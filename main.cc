#include <Grid/Grid.h>
#include <cassert>

#include <quda.h>
#include <invert_quda.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

using Coordinate = std::array<int,4>;

inline Coordinate coordinate_from_index(int index, const Coordinate& size)
{
  Coordinate x;
  x[0] = index % size[0];
  index /= size[0];
  x[1] = index % size[1];
  index /= size[1];
  x[2] = index % size[2];
  index /= size[2];
  x[3] = index % size[3];
  return x;
}

int main(int argc, char** argv) {

  const int Ls = 12;
	RealD mass = 1.0;
	RealD M5   = 1.8;
	RealD b    = 22./12.;
	RealD c    = 10./12.;
  RealD mq2  = 0.085;
  RealD mq3  = 1.0;
  RealD eofa_shift = -1.0;
  int eofa_pm = 1;

  Grid_init(&argc, &argv);
	GridLogIterative.Active(1);
	
  std::vector<int> simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
	std::vector<int> mpi_layout  = GridDefaultMpi();
	std::vector<int> latt_size   = GridDefaultLatt();
	
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
	GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
	GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
	GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

	UGrid->show_decomposition();

  initCommsGridQuda(4, mpi_layout.data(), nullptr, nullptr);
  
  printfQuda("initializing communciation on node #%03d... \n", comm_rank());
  // initQuda(UGrid->ThisRank()%4);
  initQuda(-1000);
  printfQuda("communciation initialized. \n");

  // Now setup all the QUDA parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  
  Coordinate local_dim;
  for(int mu = 0; mu < 4; mu++){
    gauge_param.X[mu] = local_dim[mu] = latt_size[mu]/mpi_layout[mu];
  }
  inv_param.Ls = Ls;
  
  gauge_param.type        = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;

  gauge_param.cpu_prec    = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.gauge_fix   = QUDA_GAUGE_FIXED_NO;

  gauge_param.anisotropy  = 1.0;
  gauge_param.t_boundary  = QUDA_PERIODIC_T;
  gauge_param.ga_pad = 0;

  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  int pad_size = std::max(x_face_size, y_face_size);
      pad_size = std::max(pad_size, z_face_size);
      pad_size = std::max(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;

  inv_param.dslash_type   = QUDA_MOBIUS_DWF_DSLASH;

  inv_param.mass          = mass;
  inv_param.m5            = M5;
  inv_param.b_5[0]        = b;
  inv_param.c_5[0]        = c;
  inv_param.mq1           = mass;
  inv_param.mq2           = mq2;
  inv_param.mq3           = mq3;
  inv_param.eofa_shift    = eofa_shift;
  inv_param.eofa_pm       = eofa_pm;
  // End setup all the QUDA parameters

//	qlat::Coordinate node_coor(UGrid->ThisProcessorCoor()[0], UGrid->ThisProcessorCoor()[1], UGrid->ThisProcessorCoor()[2], UGrid->ThisProcessorCoor()[3]);
//	qlat::Coordinate node_size(GridDefaultMpi()[0], GridDefaultMpi()[1], GridDefaultMpi()[2], GridDefaultMpi()[3]);
//	qlat::begin(qlat::index_from_coordinate(node_coor, node_size), node_size);
//	printf("Node #%03d(grid): %02dx%02dx%02dx%02d ; #%03d(qlat): %02dx%02dx%02dx%02d\n", UGrid->ThisRank(), 
//				UGrid->ThisProcessorCoor()[0], UGrid->ThisProcessorCoor()[1], UGrid->ThisProcessorCoor()[2], UGrid->ThisProcessorCoor()[3], 
//				qlat::get_id_node(), qlat::get_coor_node()[0], qlat::get_coor_node()[1], qlat::get_coor_node()[2], qlat::get_coor_node()[3]);
	
	std::vector<int> seeds4({1, 2, 3, 4});
	std::vector<int> seeds5({5, 6, 7, 8});
	GridParallelRNG RNG5(FGrid);
	RNG5.SeedFixedIntegers(seeds5);
	GridParallelRNG RNG4(UGrid);
	RNG4.SeedFixedIntegers(seeds4);

	LatticeFermion src(FGrid); gaussian(RNG5, src);
	LatticeFermion sol(FGrid); sol = zero;
	
  LatticeGaugeField Umu(UGrid);
	SU3::HotConfiguration(RNG4, Umu);

  printfQuda("Grid computed plaquette = %16.12e\n", WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu));

{
  using sobj = LatticeGaugeField::vector_object::scalar_object;
  // using sobj = vLorentzColourMatrix;
  std::vector<sobj> out_lex(Umu._grid->lSites());
  unvectorizeToLexOrdArray(out_lex, Umu);
  
  printfQuda("sizeof(sobj) = %d\n", sizeof(sobj)); 
  int V = local_dim[0]*local_dim[1]*local_dim[2]*local_dim[3];
  int Vh = V/2;
  assert(Umu._grid->lSites() == V);
  
  sobj* quda_gauge = (sobj*)malloc(Umu._grid->lSites()*sizeof(sobj));
  for(int grid_idx = 0; grid_idx < V; grid_idx++){
    Coordinate Y = coordinate_from_index(grid_idx, local_dim);
    int eo = (Y[0]+Y[1]+Y[2]+Y[3])%2;
    int quda_idx = grid_idx/2 + eo*Vh;
    quda_gauge[quda_idx] = out_lex[grid_idx];
  }
  
  loadGaugeQuda((void*)quda_gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %16.12e (spatial = %16.12e, temporal = %16.12e)\n", plaq[0], plaq[1], plaq[2]);
}

//	FieldMetaData header;
//	std::string file("/global/homes/j/jiquntu/configurations/32x64x12ID_b1.75_mh0.045_ml0.0001/configurations/ckpoint_lat.160");
//	NerscIO::readConfiguration(Umu, header, file);

	std::cout << GridLogMessage << "Lattice dimensions: " << GridDefaultLatt() << "   Ls: " << Ls << std::endl;

	MobiusEOFAFermionR DMobiusEOFA(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, mq2, mq3, eofa_shift, eofa_pm, M5, b, c);
	DMobiusEOFA.ZeroCounters();

	LatticeFermionD src_odd(FrbGrid);
	LatticeFermionD sol_odd(FrbGrid);
	pickCheckerboard(Odd, src_odd, src);
	pickCheckerboard(Odd, sol_odd, sol);
	
//	GridStopWatch CGTimer;

//	SchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD> HermOpEO(DMobiusEOFA);
//	ConjugateGradient<LatticeFermion> CG(1e-4, 2000, 0);// switch off the assert

  DMobiusEOFA.Mooee(src_odd, sol_odd);

//	LatticeFermionD mdag_src_odd(FrbGrid);
//	mdag_src_odd.checkerboard = Odd;
//	HermOpEO.AdjOp(src_odd, mdag_src_odd);
//	
//  CGTimer.Start();
//	CG(HermOpEO, mdag_src_odd, sol_odd);
//	CGTimer.Stop();
//	std::cout << GridLogMessage << "Total CG time : " << CGTimer.Elapsed() << std::endl;

  std::cout << GridLogMessage << "EOFA norm   " << norm2(src_odd) << " " << norm2(sol_odd) << std::endl;

	DMobiusEOFA.Report();

	Grid_finalize();
}
