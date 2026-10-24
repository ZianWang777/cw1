#include <nanobench.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <vmlib/stdtypes.hpp>

// Task 2.3: performance of matrix inversion compared with matrix multiplication.
//
// Build and run this in both the debug and the release configuration. Each run
// prints the results and also writes them to bench-<configuration>.csv in the
// current directory.

namespace
{
#	if defined(NDEBUG)
	char const* const kBuildName = "release";
#	else
	char const* const kBuildName = "debug";
#	endif

	std::mt19937 gRng( 1234u ); // fixed seed: the same matrices in every run

	template< std::size_t tN >
	SmallMat<tN,tN,float> random_matrix()
	{
		std::uniform_real_distribution<float> dist( -1.f, 1.f );

		SmallMat<tN,tN,float> ret{};
		for( auto& x : ret.v )
			x = dist( gRng );
		return ret;
	}

	// Inversion with Gaussian elimination
	// It is called directly so that N = 4 also uses Gaussian elimination
	// (invert() itself would pick the provided 4x4 code for Mat44f).
	template< std::size_t tN >
	SmallMat<tN,tN,float> invert_gauss( SmallMat<tN,tN,float> const& aMat )
	{
		SmallMat<tN,tN,float> ret = make_identity<SmallMat<tN,tN,float>>();
		gauss( aMat, ret );
		return ret;
	}

	// Two random test matrices per size. They are static, which keeps the large
	// matrices off the stack.
	template< std::size_t tN >
	SmallMat<tN,tN,float> const& matrix_a()
	{
		static SmallMat<tN,tN,float> const a = random_matrix<tN>();
		return a;
	}
	template< std::size_t tN >
	SmallMat<tN,tN,float> const& matrix_b()
	{
		static SmallMat<tN,tN,float> const b = random_matrix<tN>();
		return b;
	}

	template< std::size_t tN >
	void bench_invert( ankerl::nanobench::Bench& aBench )
	{
		auto const& a = matrix_a<tN>();
		aBench.run( "N=" + std::to_string( tN ), [&] {
			auto const res = invert_gauss( a );
			ankerl::nanobench::doNotOptimizeAway( res );
		} );
	}

	template< std::size_t tN >
	void bench_multiply( ankerl::nanobench::Bench& aBench )
	{
		auto const& a = matrix_a<tN>();
		auto const& b = matrix_b<tN>();
		aBench.run( "N=" + std::to_string( tN ), [&] {
			auto const res = a * b;
			ankerl::nanobench::doNotOptimizeAway( res );
		} );
	}

	template< std::size_t... tNs >
	void bench_sizes( ankerl::nanobench::Bench& aInvert, ankerl::nanobench::Bench& aMultiply )
	{
		( bench_invert<tNs>( aInvert ), ... );
		( bench_multiply<tNs>( aMultiply ), ... );
	}
}

int main()
{
	using namespace ankerl::nanobench;

	std::string const build = kBuildName;

	// Each benchmark is measured in 21 epochs (instead of 11) of at least 10 ms
	// each, and the median is reported. This makes the results more stable.
	std::size_t const epochs = 21;
	auto const minEpoch = std::chrono::milliseconds( 10 );

	// Scaling with N, up to N = 100
	Bench benchInvert;
	benchInvert.title( "invert (" + build + ")" ).unit( "inversion" ).warmup( 3 ).epochs( epochs ).minEpochTime( minEpoch );

	Bench benchMultiply;
	benchMultiply.title( "multiply (" + build + ")" ).unit( "multiplication" ).warmup( 3 ).epochs( epochs ).minEpochTime( minEpoch );

	bench_sizes<2,3,4,5,6,8,10,12,16,20,24,32,40,48,56,64,72,80,88,96,100>( benchInvert, benchMultiply );

	// 4x4 matrices in detail
	static Mat44f const a4 = random_matrix<4>();
	static Mat44f const b4 = random_matrix<4>();

	Bench bench4x4;
	bench4x4.title( "4x4 (" + build + ")" ).unit( "op" ).warmup( 100 ).epochs( epochs ).minEpochTime( minEpoch ).relative( true );

	bench4x4.run( "multiply", [&] {
		auto const res = a4 * b4;
		doNotOptimizeAway( res );
	} );
	bench4x4.run( "invert, provided 4x4 code", [&] {
		auto const res = invert( a4 );
		doNotOptimizeAway( res );
	} );
	bench4x4.run( "invert, Gaussian elimination", [&] {
		auto const res = invert_gauss( a4 );
		doNotOptimizeAway( res );
	} );

	// All results as CSV, for plotting
	std::string const csvName = "bench-" + build + ".csv";
	std::ofstream csv( csvName );
	render( templates::csv(), benchInvert, csv );
	render( templates::csv(), benchMultiply, csv );
	render( templates::csv(), bench4x4, csv );

	std::cout << "\nResults written to " << csvName << "\n";
	return 0;
}