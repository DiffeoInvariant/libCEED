// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

//                        libCEED + PETSc Example: Navier-Stokes
//
// This example demonstrates a simple usage of libCEED with PETSc to solve a
// Navier-Stokes problem.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] navierstokes
//
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -problem density_current -degree 1
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//
//TESTARGS(name="dc_explicit") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin
//TESTARGS(name="dc_implicit_stab_none") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin
//TESTARGS(name="adv_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection -strong_form 1 -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-explicit-strong.bin
//TESTARGS(name="adv_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -problem_advection_wind translation -problem_advection_wind_translation .53,-1.33,-2.65 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-translation-implicit-stab-su.bin
//TESTARGS(name="adv2d_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection2d -strong_form 1 -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-strong.bin
//TESTARGS(name="adv2d_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv2d_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -problem_advection_wind translation -problem_advection_wind_translation .53,-1.33,0 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-translation-implicit-stab-su.bin
//TESTARGS(name="euler_implicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 1,1,2 -units_meter 1e-4 -lx 125 -ly 125 -lz 1 -problem_euler_mean_velocity 1.4,-2.,0 -vortex_strength 2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-implicit.bin

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include "navierstokes.h"

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  DM dm, dmcoord;
  TS ts;
  TSAdapt adapt;
  User user;
  Units units;
  CeedData ceed_data;
  SetupContext ctxSetupData;
  Physics ctxPhysData;
  AppCtx app_ctx;
  problemData   *problem = NULL;
  PetscInt localNelemVol, lnodes, gnodes, steps;
  const PetscInt ncompq = 5;
  PetscMPIInt rank;
  PetscScalar ftime;
  Vec Q, Qloc, Xloc;
  Ceed ceed;
  CeedQFunctionContext ctxSetup, ctxNS, ctxAdvection, ctxEuler;
  PetscInt Xloc_size;
  SimpleBC bc;
  double start, cpu_time_used;

  // todo: define a function
  // Check PETSc CUDA support
  CeedMemType memtyperequested;
  PetscBool petschavecuda, setmemtyperequest = PETSC_FALSE;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*
  memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;

  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // ---------------------------------------------------------------------------
  // Allocate contexts
  // ---------------------------------------------------------------------------
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &problem); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &bc); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &ctxSetupData); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &ctxPhysData); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &app_ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &ceed_data); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // Register problems to be available on the command line
  ierr = RegisterProblems_NS(app_ctx); CHKERRQ(ierr);

  // Process general command line options
  comm = PETSC_COMM_WORLD;
  ierr = ProcessCommandLineOptions(comm, app_ctx); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(problemData *, void *, void *, void *);
    ierr = PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p);
    CHKERRQ(ierr);
    if (!p) SETERRQ1(PETSC_COMM_SELF, 1, "Problem '%s' not found",
                       app_ctx->problem_name);
    ierr = (*p)(problem, &ctxSetupData, &units, &ctxPhysData); CHKERRQ(ierr);
  }

  // todo: either complete the list here or remove it altogether.
  const CeedInt dim = problem->dim,
                ncompx = problem->dim,
                qdatasizeVol = problem->qdatasizeVol,
                numP = app_ctx->degree + 1, // Set number of 1D nodes and quadrature points
                numQ = numP + app_ctx->q_extra;
  user->phys = ctxPhysData;

  // ---------------------------------------------------------------------------
  // Setup DM
  // ---------------------------------------------------------------------------
  // Create distribute DM
  ierr = CreateDistributedDM(comm, problem, ctxSetupData, &dm); CHKERRQ(ierr);

  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = SetUpDM(dm, problem, app_ctx->degree, bc, ctxPhysData, ctxSetupData);
  CHKERRQ(ierr);

  // Refine DM for high-order viz
  user->app_ctx = app_ctx;
  if (app_ctx->viz_refine) {
    ierr = VizRefineDM(dm, user, problem, bc, ctxPhysData, ctxSetupData);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // todo
  // ---------------------------------------------------------------------------
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qloc); CHKERRQ(ierr);
  ierr = VecGetSize(Qloc, &lnodes); CHKERRQ(ierr);
  lnodes /= ncompq;

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // Initialize backend
  CeedInit(app_ctx->ceed_resource, &ceed);

// Check preferred MemType
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  // Check memtype compatibility
  if (!setmemtyperequest)
    memtyperequested = memtypebackend;
  else if (!petschavecuda && memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    CeedInt gdofs, odofs;
    int comm_size;
    char box_faces_str[PETSC_MAX_PATH_LEN] = "NONE";
    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    gnodes = gdofs/ncompq;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str,
                                 sizeof(box_faces_str), NULL); CHKERRQ(ierr);
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    ierr = PetscPrintf(comm,
                       "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                       "  rank(s)                              : %d\n"
                       "  Problem:\n"
                       "    Problem Name                       : %s\n"
                       "    Stabilization                      : %s\n"
                       "  PETSc:\n"
                       "    Box Faces                          : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (P)       : %d\n"
                       "    Number of 1D Quadrature Points (Q) : %d\n"
                       "    Global DoFs                        : %D\n"
                       "    Owned DoFs                         : %D\n"
                       "    DoFs per node                      : %D\n"
                       "    Global nodes                       : %D\n"
                       "    Owned nodes                        : %D\n",
                       comm_size, app_ctx->problem_name, StabilizationTypes[ctxPhysData->stab],
                       box_faces_str, usedresource, CeedMemTypes[memtypebackend],
                       (setmemtyperequest) ? CeedMemTypes[memtyperequested] : "none",
                       numP, numQ, gdofs, odofs, ncompq, gnodes, lnodes); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Set up global mass vector
  // ---------------------------------------------------------------------------
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xloc, &Xloc_size); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, Xloc_size, &ceed_data->xcorners); CHKERRQ(ierr);

  ierr = SetupLibceed(ceed, ceed_data, dm, user, app_ctx, problem, bc);
  CHKERRQ(ierr);

  // Set up contex for QFunctions
  CeedQFunctionContextCreate(ceed, &ctxSetup);
  CeedQFunctionContextSetData(ctxSetup, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof *ctxSetupData, ctxSetupData);

  CeedQFunctionContextCreate(ceed, &ctxNS);
  CeedQFunctionContextSetData(ctxNS, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->dc_ctx_data, ctxPhysData->dc_ctx_data);

  CeedQFunctionContextCreate(ceed, &ctxEuler);
  CeedQFunctionContextSetData(ctxEuler, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->euler_ctx_data, ctxPhysData->euler_ctx_data);

  CeedQFunctionContextCreate(ceed, &ctxAdvection);
  CeedQFunctionContextSetData(ctxAdvection, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->advection_ctx_data, ctxPhysData->advection_ctx_data);

  if (ceed_data->qf_ics && strcmp(app_ctx->problem_name, "euler_vortex") != 0)
    CeedQFunctionSetContext(ceed_data->qf_ics, ctxSetup);

  if (strcmp(app_ctx->problem_name, "density_current") == 0) {
    if (ceed_data->qf_rhsVol)
      CeedQFunctionSetContext(ceed_data->qf_rhsVol, ctxNS);

    if (ceed_data->qf_ifunctionVol)
      CeedQFunctionSetContext(ceed_data->qf_ifunctionVol, ctxNS);

  } else if (strcmp(app_ctx->problem_name, "euler_vortex") == 0) {
    if (ceed_data->qf_ics)
      CeedQFunctionSetContext(ceed_data->qf_ics, ctxEuler);

    if (ceed_data->qf_rhsVol)
      CeedQFunctionSetContext(ceed_data->qf_rhsVol, ctxEuler);

    if (ceed_data->qf_ifunctionVol)
      CeedQFunctionSetContext(ceed_data->qf_ifunctionVol, ctxEuler);

    if (ceed_data->qf_applySur)
      CeedQFunctionSetContext(ceed_data->qf_applySur, ctxEuler);
  } else {
    if (ceed_data->qf_rhsVol)
      CeedQFunctionSetContext(ceed_data->qf_rhsVol, ctxAdvection);

    if (ceed_data->qf_ifunctionVol)
      CeedQFunctionSetContext(ceed_data->qf_ifunctionVol, ctxAdvection);

    if (ceed_data->qf_applySur)
      CeedQFunctionSetContext(ceed_data->qf_applySur, ctxAdvection);
  }

  // Set up user structure
  user->comm = comm;
  user->units = units;
  user->dm = dm;
  user->ceed = ceed;


  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  ierr = VectorPlacePetscVec(ceed_data->q0ceed, Qloc); CHKERRQ(ierr);

  // Apply Setup Ceed Operators
  ierr = VectorPlacePetscVec(ceed_data->xcorners, Xloc); CHKERRQ(ierr);
  CeedOperatorApply(ceed_data->op_setupVol, ceed_data->xcorners, ceed_data->qdata,
                    CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, ceed_data->restrictq,
                                 ceed_data->basisq, ceed_data->restrictqdi, ceed_data->qdata,
                                 user->M); CHKERRQ(ierr);

  ierr = ICs_FixMultiplicity(ceed_data->op_ics, ceed_data->xcorners,
                             ceed_data->q0ceed, dm, Qloc, Q, ceed_data->restrictq,
                             ctxSetup, 0.0); CHKERRQ(ierr);
  if (1) { // Record boundary values from initial condition and override DMPlexInsertBoundaryValues()
    // We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow.  If we
    // disable this, we should still get the same results due to the problem->bc function, but with potentially much
    // slower execution.
    Vec Qbc;
    ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = VecCopy(Qloc, Qbc); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
    ierr = VecAXPY(Qbc, -1., Qloc); CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)dm,
                                      "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS);
    CHKERRQ(ierr);
  }

  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(app_ctx->output_dir); CHKERRQ(ierr);}
  // Gather initial Q values
  // In case of continuation of simulation, set up initial values from binary file
  if (app_ctx->cont_steps) { // continue from existent solution
    PetscViewer viewer;
    char file_path[PETSC_MAX_PATH_LEN];
    // Read input
    ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                         app_ctx->test_mode);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, file_path, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Q, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &Qloc); CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm); CHKERRQ(ierr);
  if (ctxPhysData->implicit) {
    ierr = TSSetType(ts, TSBDF); CHKERRQ(ierr);
    if (user->op_ifunction) {
      ierr = TSSetIFunction(ts, NULL, IFunction_NS, &user); CHKERRQ(ierr);
    } else {                    // Implicit integrators can fall back to using an RHSFunction
      ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
    }
  } else {
    if (!user->op_rhs) SETERRQ(comm, PETSC_ERR_ARG_NULL,
                                 "Problem does not provide RHSFunction");
    ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-2 * units->second); CHKERRQ(ierr);
  if (app_ctx->test_mode) {ierr = TSSetMaxSteps(ts, 10); CHKERRQ(ierr);}
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!app_ctx->cont_steps) { // print initial condition
    if (!app_ctx->test_mode) {
      ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);
    }
  } else { // continue from time of last output
    PetscReal time;
    PetscInt count;
    PetscViewer viewer;
    char file_path[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-time.bin",
                         app_ctx->test_mode); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, file_path, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(ts, time * user->units->second); CHKERRQ(ierr);
  }
  if (!app_ctx->test_mode) {
    ierr = TSMonitorSet(ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject)ts); CHKERRQ(ierr);
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(ts, &ftime); CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (!app_ctx->test_mode) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution (sec): %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }

  // Get error
  if (problem->non_zero_time && !app_ctx->test_mode) {
    Vec Qexact, Qexactloc;
    PetscReal rel_error, norm_error, norm_exact;
    ierr = DMCreateGlobalVector(dm, &Qexact); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecGetSize(Qexactloc, &lnodes); CHKERRQ(ierr);

    ierr = ICs_FixMultiplicity(ceed_data->op_ics, ceed_data->xcorners,
                               ceed_data->q0ceed, dm, Qexactloc, Qexact,
                               ceed_data->restrictq, ctxSetup, ftime); CHKERRQ(ierr);
    ierr = VecNorm(Qexact, NORM_1, &norm_exact); CHKERRQ(ierr);
    ierr = VecAXPY(Q, -1.0, Qexact);  CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_1, &norm_error); CHKERRQ(ierr);
    rel_error = norm_error / norm_exact;
    CeedVectorDestroy(&ceed_data->q0ceed);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Relative Error: %g\n",
                       (double)rel_error); CHKERRQ(ierr);
    // Clean up vectors
    ierr = DMRestoreLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecDestroy(&Qexact); CHKERRQ(ierr);
  }

  // Output Statistics
  ierr = TSGetStepNumber(ts, &steps); CHKERRQ(ierr);
  if (!app_ctx->test_mode) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps, (double)ftime); CHKERRQ(ierr);
  }
  // Output numerical values from command line
  ierr = VecViewFromOptions(Q, NULL, "-vec_view"); CHKERRQ(ierr);

  // Compare reference solution values with current test run for CI
  if (app_ctx->test_mode) {
    PetscViewer viewer;
    // Read reference file
    Vec Qref;
    PetscReal error, Qrefnorm;
    ierr = VecDuplicate(Q, &Qref); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, app_ctx->file_path, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Qref, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    // Compute error with respect to reference solution
    ierr = VecAXPY(Q, -1.0, Qref);  CHKERRQ(ierr);
    ierr = VecNorm(Qref, NORM_MAX, &Qrefnorm); CHKERRQ(ierr);
    ierr = VecScale(Q, 1./Qrefnorm); CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_MAX, &error); CHKERRQ(ierr);
    ierr = VecDestroy(&Qref); CHKERRQ(ierr);
    // Check error
    if (error > app_ctx->test_tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
                         "Test failed with error norm %g\n",
                         (double)error); CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  // Clean up libCEED
  CeedVectorDestroy(&ceed_data->qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->qdotceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&ceed_data->xcorners);
  CeedBasisDestroy(&ceed_data->basisq);
  CeedBasisDestroy(&ceed_data->basisx);
  CeedBasisDestroy(&ceed_data->basisxc);
  CeedElemRestrictionDestroy(&ceed_data->restrictq);
  CeedElemRestrictionDestroy(&ceed_data->restrictx);
  CeedElemRestrictionDestroy(&ceed_data->restrictqdi);
  CeedQFunctionDestroy(&ceed_data->qf_setupVol);
  CeedQFunctionDestroy(&ceed_data->qf_ics);
  CeedQFunctionDestroy(&ceed_data->qf_rhsVol);
  CeedQFunctionDestroy(&ceed_data->qf_ifunctionVol);
  CeedQFunctionContextDestroy(&ctxSetup);
  CeedQFunctionContextDestroy(&ctxNS);
  CeedQFunctionContextDestroy(&ctxAdvection);
  CeedQFunctionContextDestroy(&ctxEuler);
  CeedOperatorDestroy(&ceed_data->op_setupVol);
  CeedOperatorDestroy(&ceed_data->op_ics);
  CeedOperatorDestroy(&user->op_rhs_vol);
  CeedOperatorDestroy(&user->op_ifunction_vol);
  CeedDestroy(&ceed);
  CeedBasisDestroy(&ceed_data->basisqSur);
  CeedBasisDestroy(&ceed_data->basisxSur);
  CeedBasisDestroy(&ceed_data->basisxcSur);
  CeedQFunctionDestroy(&ceed_data->qf_setupSur);
  CeedQFunctionDestroy(&ceed_data->qf_applySur);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = MatDestroy(&user->interpviz); CHKERRQ(ierr);
  ierr = DMDestroy(&user->dmviz); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = PetscFree(problem); CHKERRQ(ierr);
  ierr = PetscFree(bc); CHKERRQ(ierr);
  ierr = PetscFree(ctxSetupData); CHKERRQ(ierr);
  ierr = PetscFree(ctxPhysData); CHKERRQ(ierr);
  ierr = PetscFree(app_ctx); CHKERRQ(ierr);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);
  return PetscFinalize();
}
