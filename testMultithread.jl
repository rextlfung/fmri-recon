#=
Short script for testing multithreading in Julia.
=#
using Base.Threads

function testMultithread(itrs=1000000)
	tot = zeros(itrs)

	function test(tot::AbstractArray)
	    out = zeros(size(tot))
	    for i in 1:itrs
	        out[i] = tot[i] + 1
	    end
	    return out
	end
	
	function test_thread(tot::AbstractArray)
	    out = zeros(size(tot))
	    @threads for i in 1:itrs
	        out[i] = tot[i] + 1
	    end
	    return out
	end
	
	@time out = test(tot)
	@time out_thread = test_thread(tot)

	# Validate results
	@assert out == ones(itrs)
	@assert out_thread == ones(itrs)
	
	println("Multithreading appears to be working properly")
end