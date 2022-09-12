using LinearAlgebra
using Random
using SpecialFunctions
using Hadamard

# Code here is adapted from the MATLAB code provided by Anil Damle.

"""
	get_matrix(dim::Int, k::Int, res::Float64; spectrum::String = "projector", RSvectors::String = "random", return_svd::Bool = false)

Randomly generate a square matrix `A` of dimension `dim` such that the spectral distance from `A` to its optimal rank `k` approximation
is equal to `res`. Construct the singular spectrum so that it decays according to `spectrum`. Options are:

1. "projector", a step function spectrum which decreases from 1 to `res` at index `k + 1`.
2. "smooth_gap", a smooth decrease from 1 to `res` over an interval of width `dim/3`, ending at index `k + 1`.
3. "decay", an exponential decay from 1 to `res` over the interval between 1 and `k + 1`.

Construct the right singular vectors to have the structure specified in `RSVectors`. Options are:

1. "random", an orthogonalization of a Gaussian random matrix.
2. "perm", a permutation matrix (perfectly coherent).
3. "coherent", a small perturbation of a permutation matrix.
4. "hadamard", a Hadamard matrix (perfectly incoherent).
5. "incoherent", a small perturbation of a Hadamard matrix.

If `return_svd == true`, then return `U`, `S`, `V` such that `U*diagm(S)*V'` is the SVD of the generated matrix.
"""
function get_matrix(dim::Int, k::Int, res::Float64; spectrum::String = "projector", RSvectors::String = "random", return_svd::Bool = false)
	if(dim < 1)
		throw(ErrorException("matrix dimension must be a positive integer."))
	elseif(k < 1)
		throw(ErrorException("target rank must be a positive integer."))
	elseif(k > dim)
		throw(ErrorException("target rank cannot exceed matrix dimension."))
	elseif((res <= 0) || (res >= 1))
		throw(ErrorException("target residual must be in the open interval (0, 1)."))
	end
	
	# constructing the column space (left singular subspace) of the matrix
	
	U = Matrix(qr(randn(dim, dim)).Q)
	
	# constructing the row space (right singular subspace) of the matrix
	
	if(RSvectors == "random")
		V = Matrix(qr(randn(dim, dim)).Q)
	elseif(RSvectors == "perm")
		V = Matrix{Float64}(I(dim)[:, randperm(dim)])
	elseif(RSvectors == "coherent")
		V = Matrix{Float64}(I(dim)[:, randperm(dim)])
		L, _, R = svd(V + .1*randn(dim, dim))
		V = L*R'
	elseif(RSvectors == "hadamard")
		V = hadamard(dim)/sqrt(dim)
	elseif(RSvectors == "incoherent")
		V = hadamard(dim)/sqrt(dim)
		L, _, R = svd(V + .1*randn(dim, dim))
		V = L*R'
	else
		throw(ErrorException("unsupported right singular vectors."))
	end
	
	# constructing the singular spectrum
	
	if(spectrum == "projector")
		sing_vals = ones(dim)
		
		for i = 1:k
			sing_vals[i] = 1.
		end
		
		for i = k + 1:dim
			sing_vals[i] = res
		end
	elseif(spectrum == "smooth_gap")
		decaylength = floor(Int64, .7*k)	
		x = Array(range(0., 1., dim))
		print(x)
		x .*= 5/(x[k] - x[k - decaylength])
		x .+= 2.5 - x[k]
		
		sing_vals = broadcast(t -> .5*(1 + erfc(t))/1.5, x)
		beta = log(res)/log(sing_vals[k + 1])
		sing_vals = broadcast(t -> t^beta, sing_vals)
		
		#=
		drop_length = min(k - 1, round(Int64, .3*dim))
		
		if(drop_length < 2)
			sing_vals = ones(dim)
			sing_vals[2:dim] = res*ones(dim - 1)
		else
			x = range(-1, 5, drop_length)
			x = broadcast(t -> res + .5*(1 - res)*erfc(t), x)
			
			sing_vals = zeros(dim)
			sing_vals[1:k - drop_length] = ones(k - drop_length)
			sing_vals[k - drop_length + 1:k] = x
			sing_vals[k + 1:dim] = res*ones(dim - k)
		end
		=#
	elseif(spectrum == "decay")
		rho = res^(1/k)
		sing_vals = ones(dim)
		
		for i = 2:k
			sing_vals[i] = rho*sing_vals[i - 1]
		end
		
		for i = k + 1:dim
			sing_vals[i] = res
		end
	else
		throw(ErrorException("unsupported singular spectrum."))
	end
	
	if(return_svd)
		return U, sing_vals, V
	else
		return U*diagm(sing_vals)*V'
	end
end

get_matrix(5,2,.5)
