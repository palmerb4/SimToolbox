/**
 * @file Spectral.hpp
 * @author Bryce Palmer (palme200@msu.ed)
 * @brief Utility for calculating spherical and vector spherical hamonics
 * @version 1.0
 * @date May 16, 2022
 *
 * VSH definition from FaVeST — Fast Vector Spherical Harmonic Transforms QUOC T. LE GIA
 * https://arxiv.org/pdf/1908.00041.pdf See also Quantum Theory of Angular Momentum by D A Varshalovich 1988
 */
#ifndef SPECTRAL_HPP_
#define SPECTRAL_HPP_

#include <boost/math/special_functions/spheric_harmonic.hpp>
#include <complex>

#include "EigenDef.hpp"

/*****************************
 *                           *
 * Internal Helper Functions *
 *                           *
 *****************************/

/**
 * @brief Calculate the covarient spherical basis e_{+1}
 *
 * @return e_{+1}
 */
Evec3cd _calcCovSphericalBasisP1() {
    Evec3cd e_x(1.0, 0.0, 0.0);
    Evec3cd e_y(0.0, 1.0, 0.0);
    std::complex<double> imag(0.0, 1.0);
    return -1.0 / std::sqrt(2.0) * e_x + imag * e_y;
}

/**
 * @brief Calculate the covarient spherical basis e_{0}
 *
 * @return e_{0}
 */
Evec3cd _calcCovSphericalBasis0() {
    Evec3cd e_z(0.0, 0.0, 1.0);
    return e_z;
}

/**
 * @brief Calculate the covarient spherical basis e_{-1}
 *
 * @return e_{-1}
 */
Evec3cd _calcCovSphericalBasisM1() {
    Evec3cd e_x(1.0, 0.0, 0.0);
    Evec3cd e_y(0.0, 1.0, 0.0);
    std::complex<double> imag(0.0, 1.0);
    return 1.0 / std::sqrt(2.0) * e_x - imag * e_y;
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n-1,m-1,1,1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n-1,m-1,1,1}
 */
double _clebschGordanumSpectralCoeff_nm1_mm1_1_1(const double n, const double m) {
    return std::sqrt(((n + m) * (n + m - 1)) / (2 * n * (2 * n - 1)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n+1,m-1,1,1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n+1,m-1,1,1}
 */
double _clebschGordanumSpectralCoeff_np1_mm1_1_1(const double n, const double m) {
    return std::sqrt(((n - m + 1) * (n - m + 2)) / ((2 * n + 2) * (2 * n + 3)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n-1,m,1,0}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n-1,m,1,0}
 */
double _clebschGordanumSpectralCoeff_nm1_m_1_0(const double n, const double m) {
    return std::sqrt(((n + m) * (n - m)) / (n * (2 * n - 1)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n+1,m,1,0}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n+1,m,1,0}
 */
double _clebschGordanumSpectralCoeff_np1_m_1_0(const double n, const double m) {
    return -std::sqrt(((n - m + 1) * (n + m + 1)) / ((2 * n + 3) * (n + 1)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n-1,m+1,1,-1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n-1,m+1,1,-1}
 */
double _clebschGordanumSpectralCoeff_nm1_mp1_1_m1(const double n, const double m) {
    return std::sqrt(((n - m) * (n - m - 1)) / (2 * n * (2 * n - 1)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n+1,m+1,1,-1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n+1,m+1,1,-1}
 */
double _clebschGordanumSpectralCoeff_np1_mp1_1_m1(const double n, const double m) {
    return std::sqrt(((n + m + 1) * (n + m + 2)) / ((2 * n + 3) * (2 * n + 2)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n,m-1,1,1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n,m-1,1,1}
 */
double _clebschGordanumSpectralCoeff_n_mm1_1_1(const double n, const double m) {
    return -std::sqrt(((n + m) * (n - m + 1)) / (n * (2 * n + 2)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n,m+1,1,-1}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n,m+1,1,-1}
 */
double _clebschGordanumSpectralCoeff_n_mp1_1_m1(const double n, const double m) {
    return std::sqrt(((n + m + 1) * (n - m)) / (n * (2 * n + 2)));
}

/**
 * @brief Calculate the Clebsch-Gordan (CG) coefficient C^{n, m}_{n,m,1,0}
 *
 * @param n spectral degree
 * @param m spectral order
 *
 * @return C^{n, m}_{n,m,1,0}
 */
double _clebschGordanumSpectralCoeff_n_m_1_0(const double n, const double m) { return m / std::sqrt(n * (n + 1)); }

/**
 * @brief Calculate the coefficient sqrt((n + 1) / (2n + 1))
 *
 * @param n spectral degree
 *
 * @return sqrt((n + 1) / (2n + 1))
 */
double _c(const double n) { return std::sqrt((n + 1) / (2 * n + 1)); }

/**
 * @brief Calculate the coefficient sqrt(n / (2n + 1))
 *
 * @param n spectral degree
 *
 * @return sqrt(n / (2n + 1))
 */
double _d(const double n) { return std::sqrt(n / (2 * n + 1)); }

/**
 * @brief Calculate the component of the div-free vsh in the e_{+1} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{+1} direction
 */
std::complex<double> _Bp1nm(const int n, const int m, const double theta, const double phi) {
    return _c(n) * _clebschGordanumSpectralCoeff_nm1_mm1_1_1(n, m) * calcYnm(n - 1, m - 1, theta, phi) +
           _d(n) * _clebschGordanumSpectralCoeff_np1_mm1_1_1(n, m) * calcYnm(n + 1, m - 1, theta, phi);
}

/**
 * @brief Calculate the component of the div-free vsh in the e_{0} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{0} direction
 */
std::complex<double> _B0nm(const int n, const int m, const double theta, const double phi) {
    return _c(n) * _clebschGordanumSpectralCoeff_nm1_m_1_0(n, m) * calcYnm(n - 1, m, theta, phi) +
           _d(n) * _clebschGordanumSpectralCoeff_np1_m_1_0(n, m) * calcYnm(n + 1, m, theta, phi);
}

/**
 * @brief Calculate the component of the div-free vsh in the e_{-1} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{-1} direction
 */
std::complex<double> _Bm1nm(const int n, const int m, const double theta, const double phi) {
    return _c(n) * _clebschGordanumSpectralCoeff_nm1_mp1_1_m1(n, m) * calcYnm(n - 1, m + 1, theta, phi) +
           _d(n) * _clebschGordanumSpectralCoeff_np1_mp1_1_m1(n, m) * calcYnm(n + 1, m + 1, theta, phi);
}

/**
 * @brief Calculate the component of the div-free vsh in the e_{+1} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{+1} direction
 */
std::complex<double> _Dp1nm(const int n, const int m, const double theta, const double phi) {
    return std::complex<double>(0.0, 1.0) * _clebschGordanumSpectralCoeff_n_mm1_1_1(n, m) *
           calcYnm(n, m - 1, theta, phi);
}

/**
 * @brief Calculate the component of the div-free vsh in the e_{0} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{0} direction
 */
std::complex<double> _D0nm(const int n, const int m, const double theta, const double phi) {
    return std::complex<double>(0.0, 1.0) * _clebschGordanumSpectralCoeff_n_m_1_0(n, m) * calcYnm(n, m, theta, phi);
}

/**
 * @brief Calculate the component of the div-free vsh in the e_{-1} direction
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return The component of the div-free vsh in the e_{-1} direction
 */
std::complex<double> _Dm1nm(const int n, const int m, const double theta, const double phi) {
    return std::complex<double>(0.0, 1.0) * _clebschGordanumSpectralCoeff_n_mp1_1_m1(n, m) *
           calcYnm(n, m + 1, theta, phi);
}

/********************
 *                  *
 * SH and VSH bases *
 *                  *
 ********************/

/**
 * @brief Calculate the spherical harmonics Y_n^m(theta, phi)
 *
 * @param n spectral degree
 * @param m spectral order
 * @param theta azimuthal angle
 * @param phi polar angle
 *
 * @return Y_n^m(theta, phi)
 */
std::complex<double> calcYnm(const int n, const int m, const double theta, const double phi) {
    // Boost returns zero if |m| > n
    return boost::math::spheric_harmonic(n, m, theta, phi);
}

/**
 * @brief Calculate the radial vector spherical harmonics in basis A
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaA azimuthal angle with z-axis pole
 * @param phiA polar angle with z-axis pole
 *
 * @return radial vector spherical harmonics in basis A
 */
Evec3cd getRadialVSHnmA(const int n, const int m, const double thetaA, const double phiA) {
    return calcYnm(n, m, thetaA, phiA) *
           Evec3cd(std::sin(thetaA) * std::cos(phiA), std::sin(thetaA) * std::sin(phiA), std::cos(thetaA));
}

/**
 * @brief Calculate the curl-free vector spherical harmonics in basis A
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaA azimuthal angle with z-axis pole
 * @param phiA polar angle with z-axis pole
 *
 * @return curl-free vector spherical harmonics in basis A
 */
Evec3cd getCurlfreeVSHnmA(const int n, const int m, const double thetaA, const double phiA) {
    if (n == 0) {
        Evec3cd curlfreeVSHnm(0.0, 0.0, 0.0);
    } else {
        std::complex<double> Dp1nm = _Dp1nm(n, m, thetaA, phiA);
        std::complex<double> Dm1nm = _Dm1nm(n, m, thetaA, phiA);
        std::complex<double> D0nm = _D0nm(n, m, thetaA, phiA);
        std::complex<double> imag(0.0, 1.0);

        Evec3cd curlfreeVSHnm(-1.0 / std::sqrt(2.0) * (Dp1nm - Dm1nm), -1.0 / std::sqrt(2.0) * imag * (Dp1nm + Dm1nm),
                              D0nm);
    }
    return curlfreeVSHnm
}

/**
 * @brief Calculate the div-free vector spherical harmonics in basis A
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaA azimuthal angle with z-axis pole
 * @param phiA polar angle with z-axis pole
 *
 * @return div-free vector spherical harmonics in basis A
 */
Evec3cd getDivfreeVSHnmA(const int n, const int m, const double thetaA, const double phiA) {
    if (n == 0) {
        Evec3cd divfreeVSHnm(0.0, 0.0, 0.0);
    } else {
        std::complex<double> Bp1nm = _Bp1nm(n, m, thetaA, phiA);
        std::complex<double> Bm1nm = _Bm1nm(n, m, thetaA, phiA);
        std::complex<double> B0nm = _B0nm(n, m, thetaA, phiA);
        std::complex<double> imag(0.0, 1.0);

        Evec3cd divfreeVSHnm(-1.0 / std::sqrt(2.0) * (Bp1nm - Bm1nm), -1.0 / std::sqrt(2.0) * imag * (Bp1nm + Bm1nm),
                             B0nm);
    }
    return divfreeVSHnm
}

/**
 * @brief Calculate the radial vector spherical harmonics in basis B
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaB azimuthal angle with x-axis pole
 * @param phiB polar angle with x-axis pole
 *
 * @return radial vector spherical harmonics in basis B
 */
Evec3cd getRadialVSHnmB(const int n, const int m, const double thetaB, const double phiB) {
    return calcYnm(n, m, thetaB, phiB) *
           Evec3cd(std::cos(thetaB), std::sin(thetaB) * std::sin(phiB), std::sin(thetaB) * std::cos(phiB));
}

/**
 * @brief Calculate the curl-free vector spherical harmonics in basis B
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaB azimuthal angle with x-axis pole
 * @param phiB polar angle with x-axis pole
 *
 * @return curl-free vector spherical harmonics in basis B
 */
Evec3cd getCurlfreeVSHnmB(const int n, const int m, const double thetaB, const double phiB) {
    if (n == 0) {
        Evec3cd curlfreeVSHnm(0.0, 0.0, 0.0);
    } else {
        std::complex<double> Dp1nm = _Dp1nm(n, m, thetaB, phiB);
        std::complex<double> Dm1nm = _Dm1nm(n, m, thetaB, phiB);
        std::complex<double> D0nm = _D0nm(n, m, thetaB, phiB);
        std::complex<double> imag(0.0, 1.0);

        Evec3cd curlfreeVSHnm(-1.0 / std::sqrt(2.0) * (Dp1nm - Dm1nm), -1.0 / std::sqrt(2.0) * imag * (Dp1nm + Dm1nm),
                              D0nm);
    }
    return curlfreeVSHnm
}

/**
 * @brief Calculate the div-free vector spherical harmonics in basis B
 *
 * @param n spectral degree
 * @param m spectral order
 * @param thetaB azimuthal angle with x-axis pole
 * @param phiB polar angle with x-axis pole
 *
 * @return div-free vector spherical harmonics in basis B
 */
Evec3cd getDivfreeVSHnmB(const int n, const int m, const double thetaB, const double phiB) {
    if (n == 0) {
        Evec3cd divfreeVSHnm(0.0, 0.0, 0.0);
    } else {
        std::complex<double> Bp1nm = _Bp1nm(n, m, thetaB, phiB);
        std::complex<double> Bm1nm = _Bm1nm(n, m, thetaB, phiB);
        std::complex<double> B0nm = _B0nm(n, m, thetaB, phiB);
        std::complex<double> imag(0.0, 1.0);

        Evec3cd divfreeVSHnm(-1.0 / std::sqrt(2.0) * (Bp1nm - Bm1nm), -1.0 / std::sqrt(2.0) * imag * (Bp1nm + Bm1nm),
                             B0nm);
    }
    return divfreeVSHnm
}

#endif