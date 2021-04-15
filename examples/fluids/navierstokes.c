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
  PetscInt ierr;
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  ierr = PetscCalloc1(1, &app_ctx); CHKERRQ(ierr);

  problemData *problem = NULL;
  ierr = PetscCalloc1(1, &problem); CHKERRQ(ierr);

  User user;
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);

  CeedData ceed_data;
  ierr = PetscCalloc1(1, &ceed_data); CHKERRQ(ierr);

  SimpleBC bc;
  ierr = PetscCalloc1(1, &bc); CHKERRQ(ierr);

  SetupContext ctxSetupData;
  ierr = PetscMalloc1(1, &ctxSetupData); CHKERRQ(ierr);

  Physics ctxPhysData;
  ierr = PetscMalloc1(1, &ctxPhysData); CHKERRQ(ierr);

  Units units;
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);

  user->app_ctx = app_ctx;
  user->units = units;
  user->phys = ctxPhysData;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  ierr = RegisterProblems_NS(app_ctx); CHKERRQ(ierr);

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  user->comm = comm;
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

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit(app_ctx->ceed_resource, &ceed);
  user->ceed = ceed;

  // -- Check preferred MemType
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  // -- Check memtype compatibility
  if (!setmemtyperequest)
    memtyperequested = memtypebackend;
  else if (!petschavecuda && memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // ---------------------------------------------------------------------------
  // Set up global mesh
  // ---------------------------------------------------------------------------
  // -- Create distribute DM
  DM dm;
  ierr = CreateDistributedDM(comm, problem, ctxSetupData, &dm); CHKERRQ(ierr);
  user->dm = dm;
  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);

  // -- Set up DM
  ierr = SetUpDM(dm, problem, app_ctx->degree, bc, ctxPhysData, ctxSetupData);
  CHKERRQ(ierr);

  // -- Refine DM for high-order viz
  if (app_ctx->viz_refine) {
    ierr = VizRefineDM(dm, user, problem, bc, ctxPhysData, ctxSetupData);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // -- Create Ceed coordinate vector
  Vec Xloc;
  PetscInt Xloc_size;
  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xloc, &Xloc_size); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, Xloc_size, &ceed_data->xcorners); CHKERRQ(ierr);

  // -- Set up libCEED objects
  ierr = SetupLibceed(ceed, ceed_data, dm, user, app_ctx, problem, bc);
  CHKERRQ(ierr);

  // -- Set up contex for QFunctions
  ierr = SetupContextForProblems(ceed, ceed_data, app_ctx, ctxSetupData,
                                 ctxPhysData); CHKERRQ(ierr);
  // -- Apply Setup Operator
  ierr = VectorPlacePetscVec(ceed_data->xcorners, Xloc); CHKERRQ(ierr);
  CeedOperatorApply(ceed_data->op_setupVol, ceed_data->xcorners, ceed_data->qdata,
                    CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Set up ICs
  // ---------------------------------------------------------------------------
  // -- Set up global state vector Q
  Vec Q;
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  // -- Set up local state vector Qloc
  Vec Qloc;
  ierr = DMGetLocalVector(dm, &Qloc); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(ceed_data->q0ceed, Qloc); CHKERRQ(ierr);

  // -- Fix multiplicity for ICs
  ierr = ICs_FixMultiplicity(ceed_data->op_ics, ceed_data->xcorners,
                             ceed_data->q0ceed, dm, Qloc, Q, ceed_data->restrictq,
                             ceed_data->ctxSetup, 0.0); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Set up lumped mass matrix
  // ---------------------------------------------------------------------------
  // -- Set up global mass vector
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // -- Compute lumped mass matrix
  ierr = ComputeLumpedMassMatrix(ceed, dm, ceed_data->restrictq,
                                 ceed_data->basisq, ceed_data->restrictqdi,
                                 ceed_data->qdata, user->M); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Record boundary values from initial condition
  // ---------------------------------------------------------------------------
  // -- This overrides DMPlexInsertBoundaryValues().
  //    We use this for the main simulation DM because the reference
  //    DMPlexInsertBoundaryValues() is very slow. If we disable this, we should
  //    still get the same results due to the problem->bc function, but with
  //    potentially much slower execution.
  if (1) {ierr = SetBCsFromICs_NS(dm, Q, Qloc); CHKERRQ(ierr);}

  // ---------------------------------------------------------------------------
  // Create output directory
  // ---------------------------------------------------------------------------
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(app_ctx->output_dir); CHKERRQ(ierr);}

  // ---------------------------------------------------------------------------
  // Gather initial Q values in case of continuation of simulation
  // ---------------------------------------------------------------------------
  // -- Set up initial values from binary file
  if (app_ctx->cont_steps) {
    ierr = SetupICsFromBinary(comm, app_ctx, Q); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    CeedInt gdofs, odofs;
    const PetscInt ncompq = 5;
    PetscInt lnodes, gnodes;
    const CeedInt numP = app_ctx->degree + 1,
                  numQ = numP + app_ctx->q_extra;
    int comm_size;
    char box_faces_str[PETSC_MAX_PATH_LEN] = "NONE";
    // -- Get global size
    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    gnodes = gdofs/ncompq;
    // -- Get local size
    ierr = VecGetSize(Qloc, &lnodes); CHKERRQ(ierr);
    lnodes /= ncompq;
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
  // -- Restore Qloc
  ierr = DMRestoreLocalVector(dm, &Qloc); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // TS: Create, setup, and solve
  // ---------------------------------------------------------------------------
  TS ts;
  PetscScalar ftime;   // todo: refactor this function furthur
  ierr = TSSolve_NS(dm, user, app_ctx, ctxPhysData, &Q, &ftime, &ts);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Post-processing
  // ---------------------------------------------------------------------------
  ierr = PostProcess_NS(ts, ceed_data, dm, problem, app_ctx, Q, ftime);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Destroy libCEED objects
  // ---------------------------------------------------------------------------
  // -- Vectors
  CeedVectorDestroy(&ceed_data->xcorners);
  CeedVectorDestroy(&ceed_data->qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->qdotceed);
  CeedVectorDestroy(&user->gceed);

  // -- Contexts
  CeedQFunctionContextDestroy(&ceed_data->ctxSetup);
  CeedQFunctionContextDestroy(&ceed_data->ctxNS);
  CeedQFunctionContextDestroy(&ceed_data->ctxAdvection);
  CeedQFunctionContextDestroy(&ceed_data->ctxEuler);

  // -- QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_setupVol);
  CeedQFunctionDestroy(&ceed_data->qf_ics);
  CeedQFunctionDestroy(&ceed_data->qf_rhsVol);
  CeedQFunctionDestroy(&ceed_data->qf_ifunctionVol);
  CeedQFunctionDestroy(&ceed_data->qf_setupSur);
  CeedQFunctionDestroy(&ceed_data->qf_applySur);

  // -- Bases
  CeedBasisDestroy(&ceed_data->basisq);
  CeedBasisDestroy(&ceed_data->basisx);
  CeedBasisDestroy(&ceed_data->basisxc);
  CeedBasisDestroy(&ceed_data->basisqSur);
  CeedBasisDestroy(&ceed_data->basisxSur);
  CeedBasisDestroy(&ceed_data->basisxcSur);

  // -- Restrictions
  CeedElemRestrictionDestroy(&ceed_data->restrictq);
  CeedElemRestrictionDestroy(&ceed_data->restrictx);
  CeedElemRestrictionDestroy(&ceed_data->restrictqdi);

  // -- Operators
  CeedOperatorDestroy(&ceed_data->op_setupVol);
  CeedOperatorDestroy(&ceed_data->op_ics);
  CeedOperatorDestroy(&user->op_rhs_vol);
  CeedOperatorDestroy(&user->op_ifunction_vol);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);

  // -- Ceed
  CeedDestroy(&ceed);

  // ---------------------------------------------------------------------------
  // Clean up PETSc
  // ---------------------------------------------------------------------------
  // -- Vectors
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);

  // -- Matrices
  ierr = MatDestroy(&user->interpviz); CHKERRQ(ierr);

  // -- DM
  ierr = DMDestroy(&user->dmviz); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  // -- TS
  ierr = TSDestroy(&ts); CHKERRQ(ierr);

  // -- Structs
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
