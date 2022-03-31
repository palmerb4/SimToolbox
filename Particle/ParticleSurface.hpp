/**
 * @file ParticleSurface.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Sphero-cylinder type
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef PARTICLESURFACE_HPP_
#define PARTICLESURFACE_HPP_

#include "Util/EigenDef.hpp"
#include "SharedParticleSurface.hpp"
#include <math>
/**
 * @brief Particle surface class 
 * Contains all the necessary information for creating and manipulating surface quantities on a particle
 * Wraps the precomputed particle surface
 * 
 * To ensure that the class remain trivial copyable, spectralDegree must be a template parameter and SharedPS must be specially handled
 */

// TODO
//   1. (done) Change all relevent functions to take in pointers rather than eigen objects, then map those pointers to Eigen objects!
//    Assumes a structure of the given object
//   2. (done) Change all relevent functions to output void and modify an input value instead
//   3. ensure that the structure of the input functions is as expected!


template <int spectralDegree>
class ParticleSurface {

  public:
    string name;
    Equatn orientation;
    std::reference_wrapper<const SharedParticleSurface<spectralDegree>> SharedPS; // reference_wrapper is a reassignable, trivially copyable reference

    // compile time variables
    // Assume Guass Legandre quadrature (with poles)
    constexpr int numGridPts = (spectralDegree + 1) * (2 * spectralDegree + 2) + 2;
    constexpr int numSpectralCoeff = std::pow(spectralDegree + 1, 2);

    // useful Eigen functionality
    using EmatNP = Eigen::Matrix<double, numGridPts, 1, Eigen::DontAlign>;
    using EmatmapNP = Eigen::Map<EmatNP, Eigen::Unaligned>;
    using ECmatmapNP = Eigen::Map<const EmatNP, Eigen::Unaligned>;
    using EmatNS = Eigen::Matrix<double, numSpectralCoeff, 1, Eigen::DontAlign>;
    using EmatmapNS = Eigen::Map<EmatNS, Eigen::Unaligned>;
    using ECmatmapNS = Eigen::Map<const EmatNS, Eigen::Unaligned>;

    using EmatNP3 = Eigen::Matrix<double, numGridPts, 3, Eigen::DontAlign>;
    using EmatmapNP3 = Eigen::Map<EmatNP, Eigen::Unaligned>;
    using ECmatmapNP3 = Eigen::Map<const EmatNP, Eigen::Unaligned>;
    using EmatNS3 = Eigen::Matrix<double, numSpectralCoeff, 3, Eigen::DontAlign>;
    using EmatmapNS3 = Eigen::Map<EmatNS, Eigen::Unaligned>;
    using ECmatmapNS3 = Eigen::Map<const EmatNS, Eigen::Unaligned>;

    // surface data
    EmatNP3 surfaceDensityGridVals;
    EmatNP3 surfaceVelocityGridVals;
    EmatNS3 surfaceDensityCoeff;
    EmatNS3 surfaceVelocityCoeff;

    /**
     * @brief Clear the mutable data strictures
     */
    void clear();

    /**
     * @brief Store a pointer to a SharedParticleSurface instance
     * 
     */
    void storeSharedSurface(const SharedParticleSurface<spectralDegree> *SharedPSPtr);

    /**
     * @brief Get the number of grid points on the surface
     * 
     * @return Number of grid points on the surface
     */
    const int getNumGridPts() const;

    /**
     * @brief Get the number of spectral coefficients used in the discretization of the surface
     *  This class must return a const expression, as it is used to initialize various arrays
     * 
     * @return Spectral coefficients used in the discretization of the surface
     */
    const int getNumSpectralCoeff() const;

    /**
     * @brief Get the gridPoint of the surface in the current configuration
     *
     * @param coordBase The coordinate of the particle's core, not necessarily its centroid
     * @param idx The index of the grid point to fetch
     *  
     * @return The gridPoint of the surface in the current configuration
     */
    const Evec3 getGridPointCurrentConfig(const Evec3 coordBase, const int idx) const;

    /**
     * @brief Get the normalized normal vector of the surface in the current configuration
     *
     * @param coordBase The coordinate of the particle's core, not necessarily its centroid
     * @param idx The index of the grid point to fetch
     *  
     * @return The normalized normal vector of the surface in the current configuration
     */
    const Evec3 getGridNormCurrentConfig(const int idx) const;

    /**
     * @brief Get the moment of inertia tensor of the surface in the current configuration
     * 
     * @return The moment of inertia tensor of the surface in the current configuration
     */
    const Emat3 getInvMomentOfInertiaTensorCurrentConfig() const;

    /**
     * @brief Get the grid weight corresponding to idx'th quadrature point
     *
     * @param idx The index of the grid point to fetch
     *  
     * @return The idx'th grid weight
     */
    const double getGridWeight(const int idx) const;

    /**
     * @brief Get the surface area of the object
     *  
     * @return The surface area of the object
     */
    const double getSurfaceArea() const;

    /**
     * @brief Compute the spectral decomposition of a surface vector evaluated at each grid point
     *
     * @param vecSurface A surface vector evaluated at each grid point 
     *  
     * @return VSHcoeff The vector spherical harmonic coefficients 
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    void decomposeSurfaceVectorFcn(const double *vecSurface, std::complex<double> *VSHcoeff) const;

    /**
     * @brief Compute the reconstruction of a surface vector based on its vector spherical hamonic coefficients
     *
     * @param VSHcoeff The vector spherical harmonic coefficients
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     *  
     * @return The reconstructed surface vector evaluated at each grid point
     */
    void reconstructSurfaceVectorFcn(const std::complex<double> *VSHcoeff, double *vecSurface) const;

    /**
     * @brief Compute the spectral decomposition of a surface scalar evaluated at each grid point
     *
     * @param scalarSurface A surface scalar function evaluated at each grid point 
     *  
     * @return The spherical harmonic coefficients of the decomposition
     */
    void decomposeSurfaceScalarFcn(const double *scalarSurface, std::complex<double> *SHcoeff) const;

    /**
     * @brief Compute the reconstruction of a surface scalar function based on its spherical hamonic coefficients
     *
     * @param SHcoeff The spherical harmonic coefficients of the decomposition 
     *  
     * @return The reconstructed surface scalar function evaluated at each grid point
     */
    void reconstructSurfaceScalarFcn(const std::complex<double> *SHcoeff, double *scalarSurface) const;

    /**
     * @brief Dump the surface to an unstructured grid
     * Each surface dumps a 'piece' where points in different pieces are completely independent
     * Although, variable names in different pieces must be the same
     * 
     * @param file The VTK file to be wrote to
     * @param coordBase The coordinate of the particle's core, not necessarily its centroid
     */
    int writeVTU(std::ofstream &file, const Evec3 &coordBase = Evec3::Zero()) const;

    /**
     * @brief Compute the cell connectivity of the grid points to be used by VTK
     *
     * @param gridCellConnect The cell connectivity of the grid points to be used by VTK
     * @param offset The offset of each cell to be used by VTK
     * @param cellTypes The VTK cell type of each cell
     */
    void calcGridCellConnect(std::vector<int32_t> &gridCellConnect, std::vector<int32_t> &offset,
                                    std::vector<uint8_t> &cellTypes) const;

    /**
     * @brief Get a user defined surface density from a given csv file
     *
     * @param prescribedSurfaceDensityFile Filename of the csv containing the surface density coefficients
     *  
     * @return The vector spherical harmonic coefficients of the decomposed surface density. 
     *          (idx 0: radial, idx 1: divFree, idx 2: curlFree)
     */
    EmatNS3 getUserInputSurfaceDensity(const string prescribedSurfaceDensityFile) const;

};

// Include the ParticleSurface implimentation
#include "ParticleSurface.tpp"

#endif
