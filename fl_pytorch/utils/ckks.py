#!/usr/bin/env python3

# Cheon-Kim-Kim-Song (CKKS) scheme (https://eprint.iacr.org/2016/421.pdf)

# Benefits:
# 1. CKKS allows us to perform HE computations on vectors of complex and real values.

# Background:

# (a) Ring - algebra in which (A,+) is Abel group and (A,·) is monoid. I.e. multiplication can be non-invertible. Multiplication is distributive w.r.t. to summation. 
#     In Ring without extra assumptions can have situation such that a != 0, b !=0, but a*b=0 
#
# (b) Let K be some field. The set of all polynomials with coefficients in K forms a commutative ring, denoted by K[x] and called the polynomial ring over k.
#     Here polynomial is usual polynomial in form p(x) = p0 + p1*x + p2*x^2 + ....

#  Remark: In any finite fields F_p it's well known that x^p = 1 (this  result hold for any finite group with number of elements in group equal to "p", e.g. in Z_p number of elements is p-1 and so a^{p-1}=1 \forall a \in Z_p).

# (c) If consider polynimials as objects they have a basis: 1, x, x^2, ....

# (d) In the Ring of polynomials P[X] there is no division in general similar for Algebra N = ({0,1,2,3,...}, +, *). However it's possible in the same way as it can be done for N perform division with residual.

# (e) Integer polynomial rings over Commutative Ring Z is denoted as Z[x]. I.e. this is the set of polynomials whose coefficients are integers and there is only variable for polynomial.

# (f) P[x]/(x^2 + 1). This specific notation from Discrete Math. It means take the polynomial ring P[x] as above, and “mod out” by the polynomial x^2 + 1.
#     Example: (x+1)^2 / (x^2+1) = x^2 + 1 + 2x / x^2 + 1 = 2x + 0 = 2x.
#     This modulus arithmetic restricts obtained polynomials to be limited by power "2".
#     This is an example of a quotient ring.


# (g) In mathematics, the greatest common divisor (GCD) of two or more integers, which are not all zero, is the largest positive integer that divides each of the integers. 
# For two integers x, y, the greatest common divisor of x and y is denoted gcd(x,y). For example gcd(8, 12) = 4.

# (f) Cyclotomic Polynomial is polynomial with all roots: exp(2i\pi kn), where k \in {1,2,...,n} AND k is co-prime with n.

# (g) Roots of unity are roots of Z^n=1, when Z is complex. The roots are: exp(2 \pi k i / n ), k=0,1,2,...,n-1

#===============================================================================================================

# Step-1: [m] a vector of values on which we want to perform certain computation, is first encoded into a plaintext polynomial [p(X)].
# This encoder step is necessary because the encryption, decryption, and other mechanisms work on polynomial rings. 
# Therefore it is necessary to have a way to transform our vectors of real values into polynomials.

# Step-2: [p(x)] is encrypted using a public key into two polynomials [c0, c1]
# Step-3: CKKS provide ways to do Add, Mult and rotate. Collectively this operations are denoted by [f(c)] and c'=f(c)
# Step-4: Decrypting [c'] with the secret key will yield [p’= f(p)]. 
# Step-5: Decode it into m'=f(m)

# The central idea to implement a homomorphic encryption scheme is to have homomorphic properties on the encoder, decoder, encryptor and decryptor.

# Assumptions:
# 1. N is a power of "2" and M=2N.
# 2. Input z \in C^{N/2} 

import numpy as np
# python - m pip install fastcore numpy
from numpy.polynomial import Polynomial
from fastcore.foundation import patch_to

def gcd(p, q):
    if q == 0:
        return p
    r = p % q;
    return gcd(q, r)

if __name__ == "__main__":
    print("GCD of 8 and 12 is 4:", gcd(8,12))
    print("GCD of 7*3 and 5*2 is 1:", gcd(7*3, 5*2))


# First we set the parameters
M = 8
N = M //2

class CKKSEncoder_:
    """Basic CKKS encoder to encode complex vectors into polynomials."""
    
    def __init__(self, M: int):
        """Initialization of the encoder for M a power of 2. 
        
        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        
    @staticmethod
    def vandermonde(xi: np.complex128, M: int) -> np.array:
        """Computes the Vandermonde matrix from a m-th root of unity."""
        
        N = M //2
        matrix = []
        # We will generate each row of the matrix
        for i in range(N):
            # For each row we select a different root
            root = xi ** (2 * i + 1)
            row = []

            # Then we store its powers
            for j in range(N):
                row.append(root ** j)
            matrix.append(row)
        return matrix
    
    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        # First we create the Vandermonde matrix
        A = CKKSEncoder_.vandermonde(self.xi, M)

        # Then we solve the system
        coeffs = np.linalg.solve(A, b)

        # Finally we output the polynomial
        p = Polynomial(coeffs)
        return p

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M //2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = p(root)
            outputs.append(output)
        return np.array(outputs)


encoder = CKKSEncoder_(M)              # First we initialize our encoder
b = np.array([1, 2, 3, 4])             # Plain input
p = encoder.sigma_inverse(b)           # Encode input vector
b_reconstructed = encoder.sigma(p)     # Decode input vector

print("TEST 1: ",np.linalg.norm(b_reconstructed - b))

m1 = np.array([1, 2, 3, 4])
m2 = np.array([1, -2, 3, -4])

p1 = encoder.sigma_inverse(m1)
p2 = encoder.sigma_inverse(m2)
p_add = p1 + p2
m_rec = encoder.sigma(p_add)

print("TEST 2: ", np.linalg.norm(m1 + m2 - m_rec))
poly_modulo = Polynomial([1, 0, 0, 0, 1])     # x-> 1.0+0.0x+0.0x^2+0.0x^3+1.0x^4 (X^N + 1)

p_mult = (p1 * p2) % poly_modulo
m_mult_rec = encoder.sigma(p_mult)

print("TEST 3: ", np.linalg.norm(m1 * m2 - m_mult_rec))


#===================================================================================================================


class CKKSEncoder:
    """Basic CKKS encoder to encode complex vectors into polynomials."""
    
    def __init__(self, M: int, scale:float):
        """Initialization of the encoder for M a power of 2. 
        
        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        self.create_sigma_R_basis()
        self.scale = scale                      # Scale to achieve a fixed level of precisio


    def pi(self, z: np.array) -> np.array:
        """Projects a vector of H into C^{N/2}."""
    
        N = self.M // 4
        return z[:N]

    def pi_inverse(self, z: np.array) -> np.array:
        """Expands a vector of C^{N/2} by expanding it with its  complex conjugate."""
    
        z_conjugate = z[::-1]
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])

    def create_sigma_R_basis(self):
        """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""
        self.sigma_R_basis = np.array(self.vandermonde(self.xi, self.M)).T


    def compute_basis_coordinates(self, z):
        """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
        output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
        return output

    @staticmethod
    def round_coordinates(coordinates):
        """Gives the integral rest."""
        coordinates = coordinates - np.floor(coordinates)
        return coordinates

    @staticmethod
    def coordinate_wise_random_rounding(coordinates):
        """Rounds coordinates randonmly."""
        r = CKKSEncoder.round_coordinates(coordinates)
        f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c]) for c in r]).reshape(-1)
    
        rounded_coordinates = coordinates - f
        rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
        return rounded_coordinates

    def sigma_R_discretization(self, z):
        """Projects a vector on the lattice using coordinate wise random rounding."""
        coordinates = self.compute_basis_coordinates(z)
    
        rounded_coordinates = CKKSEncoder.coordinate_wise_random_rounding(coordinates)
        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y

    @staticmethod
    def vandermonde(xi: np.complex128, M: int) -> np.array:
        """Computes the Vandermonde matrix from a m-th root of unity."""
        
        N = M //2
        matrix = []
        # We will generate each row of the matrix
        for i in range(N):
            # For each row we select a different root
            root = xi ** (2 * i + 1)
            row = []

            # Then we store its powers
            for j in range(N):
                row.append(root ** j)
            matrix.append(row)
        return matrix
    
    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        # First we create the Vandermonde matrix
        A = CKKSEncoder.vandermonde(self.xi, M)

        # Then we solve the system
        coeffs = np.linalg.solve(A, b)

        # Finally we output the polynomial
        p = Polynomial(coeffs)
        return p

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M //2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = p(root)
            outputs.append(output)
        return np.array(outputs)

    def encode(self, z: np.array) -> Polynomial:
        """Encodes a vector by expanding it first to H,
        scale it, project it on the lattice of sigma(R), and performs
        sigma inverse.
        """
        pi_z = self.pi_inverse(z)
        scaled_pi_z = self.scale * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p = self.sigma_inverse(rounded_scale_pi_zi)
    
        # We round it afterwards due to numerical imprecision
        coef = np.round(np.real(p.coef)).astype(int)
        p = Polynomial(coef)
        return p

    def decode(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by removing the scale, 
        evaluating on the roots, and project it on C^(N/2)
        """
        rescaled_p = p / self.scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z

scale = 6400
encoder = CKKSEncoder(M, scale)        # First we initialize our encoder

z = np.array([3 +4j, 2 - 1j])
p = encoder.encode(z)
print(p)
zr = encoder.decode(p)

print("TEST 4: ",np.linalg.norm(zr - z))

# python -m pip install syft
# https://blog.openmined.org/ckks-homomorphic-encryption-pytorch-pysyft-seal/
