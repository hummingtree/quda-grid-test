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

  double conversion_factor = (b*(4-M5)+1);

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

  inv_param.dslash_type   = QUDA_MOBIUS_DWF_EOFA_DSLASH;

  inv_param.mass          = mass;
  inv_param.m5            = -M5;
  for(int s = 0; s < Ls; s++){
  inv_param.b_5[s]        = b;
  inv_param.c_5[s]        = c;
  }
  inv_param.mq1           = mass;
  inv_param.mq2           = mq2;
  inv_param.mq3           = mq3;
  inv_param.eofa_shift    = eofa_shift;
  inv_param.eofa_pm       = eofa_pm;
  // End setup all the QUDA parameters
  
  inv_param.solve_type    = QUDA_NORMOP_PC_SOLVE;
  inv_param.matpc_type    = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger        = QUDA_DAG_NO;

  inv_param.cpu_prec      = QUDA_DOUBLE_PRECISION;;
  inv_param.cuda_prec     = QUDA_DOUBLE_PRECISION;;

  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.verbosity     = QUDA_DEBUG_VERBOSE;

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
  
  int V = local_dim[0]*local_dim[1]*local_dim[2]*local_dim[3];
  int Vh = V/2;
  void* quda_gauge = nullptr;
{
  using sobj = LatticeGaugeField::vector_object::scalar_object;
  // using sobj = vLorentzColourMatrix;
  std::vector<sobj> out_lex(Umu._grid->lSites());
  unvectorizeToLexOrdArray(out_lex, Umu);
  
  printfQuda("sizeof(sobj) = %d\n", sizeof(sobj)); 
  assert(Umu._grid->lSites() == V);
  
  quda_gauge = reinterpret_cast<void*>(malloc(Umu._grid->lSites()*sizeof(sobj)));
  for(int grid_idx = 0; grid_idx < V; grid_idx++){
    Coordinate Y = coordinate_from_index(grid_idx, local_dim);
    int eo = (Y[0]+Y[1]+Y[2]+Y[3])%2;
    int quda_idx = grid_idx/2 + eo*Vh;
    reinterpret_cast<sobj*>(quda_gauge)[quda_idx] = out_lex[grid_idx];
  }
  
  loadGaugeQuda((void*)quda_gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %16.12e (spatial = %16.12e, temporal = %16.12e)\n", plaq[0], plaq[1], plaq[2]);
}

	LatticeFermionD src_e(FrbGrid);
	LatticeFermionD sol_e(FrbGrid);
	LatticeFermionD sol_e_quda(FrbGrid);
	pickCheckerboard(Even, src_e, src);
	pickCheckerboard(Even, sol_e, sol);
	pickCheckerboard(Even, sol_e_quda, sol);

  void* quda_src = nullptr;
  void* quda_sol = nullptr;
  using fsobj = LatticeFermion::vector_object::scalar_object;

  std::vector<fsobj> out_lex(src_e._grid->lSites());
  unvectorizeToLexOrdArray(out_lex, src_e);

  assert(src_e._grid->lSites() == Vh*Ls);

  quda_src = reinterpret_cast<void*>(malloc(src_e._grid->lSites()*sizeof(fsobj)));
  quda_sol = reinterpret_cast<void*>(malloc(src_e._grid->lSites()*sizeof(fsobj)));
  
  for(int x_cb = 0; x_cb < Vh; x_cb++){
    for(int s = 0; s < Ls; s++){
      int grid_idx = x_cb*Ls+s;
      int quda_idx = s*Vh+x_cb;
      reinterpret_cast<fsobj*>(quda_src)[quda_idx] = out_lex[grid_idx];
    } 
  }

//	FieldMetaData header;
//	std::string file("/global/homes/j/jiquntu/configurations/32x64x12ID_b1.75_mh0.045_ml0.0001/configurations/ckpoint_lat.160");
//	NerscIO::readConfiguration(Umu, header, file);

	std::cout << GridLogMessage << "Lattice dimensions: " << GridDefaultLatt() << "   Ls: " << Ls << std::endl;

	MobiusEOFAFermionR DMobiusEOFA(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, mq2, mq3, eofa_shift, eofa_pm, M5, b, c);
	DMobiusEOFA.ZeroCounters();
	
//	GridStopWatch CGTimer;

//	SchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD> HermOpEO(DMobiusEOFA);
//	ConjugateGradient<LatticeFermion> CG(1e-4, 2000, 0);// switch off the assert

  dslashQuda_mobius_eofa(quda_sol, quda_src, &inv_param, QUDA_EVEN_PARITY, 0);
  std::vector<fsobj> in_lex(src_e._grid->lSites());
  for(int x_cb = 0; x_cb < Vh; x_cb++){
    for(int s = 0; s < Ls; s++){
      int grid_idx = x_cb*Ls+s;
      int quda_idx = s*Vh+x_cb;
      in_lex[grid_idx] = reinterpret_cast<fsobj*>(quda_sol)[quda_idx];
    } 
  }
  vectorizeFromLexOrdArray(in_lex, sol_e_quda);

  DMobiusEOFA.Mooee(src_e, sol_e);

  LatticeFermion err(FrbGrid);
  pickCheckerboard(Even, err, sol);
  err = sol_e - conversion_factor * sol_e_quda;

  printf("EOFA: Grid source norm2 = %16.12e, solution norm2 = %16.12e, error norm2 = %16.12e\n", norm2(src_e), norm2(sol_e), norm2(err));

	DMobiusEOFA.Report();

  endQuda();

	Grid_finalize();
}
