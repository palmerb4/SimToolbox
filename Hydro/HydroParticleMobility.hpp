#ifndef HydroParticleMobility_HPP_
#define HydroParticleMobility_HPP_

#include "HydroParticleOperator.hpp"
#include "Trilinos/Preconditioner.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Util/IOHelper.hpp"

#include "Teuchos_YamlParameterListHelpers.hpp"

#include <string>

// TODO: 
//  1. apply needs updated to match the wrapper syntax.

template <class Container>
class HydroParticleMobility : public TOP {
  private:
    const Container *const containerPtr;    ///< read-only
    const int nParticleLocal;               ///< local number of particles
    std::shared_ptr<stkfmm::STKFMM> fmmPtr; ///< pointer to fmm, either 3d or above wall
    const Config &cellConfig;               ///< hydro user input
    const SylinderConfig &runConfig;        ///< general user input

    //
    Teuchos::RCP<const TCOMM> commRcp;
    Teuchos::RCP<TMAP> mobMapRcp; ///< mobility map, 6 dof per particle

    // linear problem Ax = b
    Teuchos::RCP<HydroParticleOperator<Container>> AOpRcp;
    Teuchos::RCP<TV> xRcp;
    Teuchos::RCP<TV> bRcp;

    Teuchos::RCP<Belos::SolverManager<TOP::scalar_type, TMV, TOP>> solverRcp;
    Teuchos::RCP<Belos::LinearProblem<::TOP::scalar_type, TMV, TOP>> problemRcp;
    Teuchos::RCP<Teuchos::ParameterList> solverParamsRcp;

    void setSolverParameters();

  public:
    HydroParticleMobility(const Container *const containerPtr_, const int nParticleLocal_,
                        const SylinderConfig &runConfig_);

    ~HydroParticleMobility() = default;

    Teuchos::RCP<const TMAP> getDomainMap() const { return mobMapRcp; }

    Teuchos::RCP<const TMAP> getRangeMap() const { return mobMapRcp; }

    bool hasTransposeApply() const { return false; }

    // Compute Y := alpha Op X + beta Y.
    void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const;

    void cacheResults(Container &particleContainer) const;

    // motion generated by swim and imposed force in hydrotable
    void calcMotion(TV &VelOmegaRcp);

    void setInitialGuess(Teuchos::RCP<TV> &feGuessRcp);

    Teuchos::RCP<HydroParticleOperator<Container>> getHydroOperator() const { return AOpRcp; };

    Teuchos::RCP<TV> getfeRcp() const { return xRcp; };

    Teuchos::RCP<TV> getfsRcp() const { return AOpRcp->getfsRcp(); };

    // self test
    void testOperator();
};

// Include the HydroParticleMobility implimentation
#include "HydroParticleMobility.tpp"

#endif