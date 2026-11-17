// Opt-in benchmark for activity optimization C. No production algorithm is implemented here.
#include "support/ThbActivityChecks.hpp"
#include <NavierStokesDiscretization.hpp>
#include <RefinementPolicy.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

namespace
{
    using Clock = std::chrono::steady_clock;

    double seconds( const Clock::time_point a, const Clock::time_point b )
    {
        return std::chrono::duration<double>( b - a ).count();
    }

    std::optional<uint64_t> peakRssBytes()
    {
#if defined(__APPLE__) || defined(__linux__)
        rusage usage{};
        if( getrusage( RUSAGE_SELF, &usage ) != 0 ) return std::nullopt;
#if defined(__APPLE__)
        return static_cast<uint64_t>( usage.ru_maxrss );
#else
        return static_cast<uint64_t>( usage.ru_maxrss ) * 1024;
#endif
#else
        return std::nullopt;
#endif
    }

    std::string jsonString( const std::string& text )
    {
        std::ostringstream out;
        out << '"';
        for( const unsigned char ch : text )
        {
            if( ch == '"' or ch == '\\' ) out << '\\' << ch;
            else if( ch < 32 ) out << "\\u" << std::hex << std::setw( 4 ) << std::setfill( '0' ) << unsigned( ch );
            else out << ch;
        }
        return out.str() + '"';
    }

    std::vector<double> openKnots( const size_t n )
    {
        std::vector<double> knots( 3, 0.0 );
        for( size_t i = 0; i <= n; i++ ) knots.push_back( static_cast<double>( i ) );
        knots.insert( knots.end(), 3, static_cast<double>( n ) );
        return knots;
    }

    Eigen::MatrixXd controlPoints( const std::vector<double>& knots, const double x0, const bool homogeneous )
    {
        std::vector<double> greville;
        for( size_t i = 0; i + 4 < knots.size(); i++ )
            greville.push_back( ( knots.at( i + 1 ) + knots.at( i + 2 ) + knots.at( i + 3 ) ) / 3.0 );
        Eigen::MatrixXd points( homogeneous ? 3 : 2, greville.size() * greville.size() );
        for( size_t j = 0; j < greville.size(); j++ )
            for( size_t i = 0; i < greville.size(); i++ )
            {
                const size_t col = i + greville.size() * j;
                points( 0, col ) = x0 + greville.at( i );
                points( 1, col ) = greville.at( j );
                if( homogeneous ) points( 2, col ) = 1.0;
            }
        return points;
    }

    size_t extractionNnz( const basis::HierarchicalTPSplineSpace& space )
    {
        size_t sum = 0;
        for( const auto& op : space.levelExtractionOperators() ) sum += op.nonZeros();
        return sum;
    }

    size_t extractionNnz( const basis::HierarchicalMultiPatchSplineSpace& space )
    {
        size_t sum = 0;
        for( const auto& patch : space.constituents() ) sum += extractionNnz( *patch );
        return sum;
    }

    size_t extractionNnz( const basis::VectorConformingHierarchicalTPSplineSpace& space )
    {
        size_t sum = 0;
        for( const auto& scalar : space.scalarBases() ) sum += extractionNnz( *scalar );
        return sum;
    }

    size_t extractionNnz( const basis::VectorConformingHierarchicalMultiPatchSplineSpace& space )
    {
        size_t sum = 0;
        for( const auto& patch : space.constituents() ) sum += extractionNnz( *patch );
        return sum;
    }

    template <typename Build>
    void measure( const std::string& fixture, const size_t depth, const size_t expected_leaves, Build build )
    {
        const auto start = Clock::now();
        const auto disc = build();
        const auto constructed = Clock::now();
        const auto ctor_rss = peakRssBytes();
        const auto policy_start = Clock::now();
        api::requireRefinementPolicy( *disc, api::RefinementPolicy::NFormSupport );
        const auto policy_end = Clock::now();
        thb_test::ActivityReport activity;
        thb_test::checkActivity( disc->H1_ss, "H1", activity );
        thb_test::checkActivity( disc->HDIV_ss, "Hdiv", activity );
        thb_test::checkActivity( disc->L2_ss, "L2", activity );
        using Discretization = typename decltype( disc )::element_type;
        constexpr bool single = std::is_base_of_v<api::NavierStokesHierarchicalDiscretization, Discretization>;
        if( activity.levels != ( single ? 4 : 3 ) * ( depth + 1 ) )
            throw std::runtime_error( "Not every activity level was compared" );
        size_t leaves = 0;
        for( const auto& level : leafElements( disc->H1_ss.basisComplex().parametricAtlas().cmap() ) )
            leaves += level.size();
        if( leaves != expected_leaves ) throw std::runtime_error( "Benchmark fixture leaf count changed" );
        const size_t nnz = extractionNnz( disc->H1_ss ) + extractionNnz( disc->HDIV_ss ) + extractionNnz( disc->L2_ss );
        const auto final_rss = peakRssBytes();
#if defined(__clang__)
        const std::string compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
        const std::string compiler = "gcc " __VERSION__;
#else
        const std::string compiler = "unknown";
#endif
        std::cout << std::setprecision( 17 )
                  << "{\"schema\":1,\"fixture\":" << jsonString( fixture ) << ",\"depth\":" << depth
                  << ",\"degree\":3,\"verified\":true,\"leaves\":" << leaves
                  << ",\"h1_functions\":" << disc->H1_ss.numFunctions()
                  << ",\"hdiv_functions\":" << disc->HDIV_ss.numFunctions()
                  << ",\"l2_functions\":" << disc->L2_ss.numFunctions()
                  << ",\"extraction_nnz\":" << nnz
                  << ",\"constructor_s\":" << seconds( start, constructed )
                  << ",\"policy_s\":" << seconds( policy_start, policy_end )
                  << ",\"reference_s\":" << activity.reference_seconds
                  << ",\"production_activity_s\":" << activity.production_seconds
                  << ",\"compared_levels\":" << activity.levels
                  << ",\"activity_fingerprint\":" << jsonString( std::to_string( activity.fingerprint ) )
                  << ",\"compiler\":" << jsonString( compiler )
                  << ",\"compiler_path\":" << jsonString( THB_BENCH_COMPILER )
                  << ",\"build_type\":" << jsonString( THB_BENCH_BUILD_TYPE )
                  << ",\"pointer_bits\":" << 8 * sizeof( void* )
                  << ",\"peak_rss_after_constructor_bytes\":";
        if( ctor_rss ) std::cout << *ctor_rss; else std::cout << "null";
        std::cout << ",\"peak_rss_after_checks_bytes\":";
        if( final_rss ) std::cout << *final_rss; else std::cout << "null";
        std::cout << "}\n";
    }

    void run( const std::string& fixture, const size_t depth )
    {
        if( fixture == "F2" )
        {
            const auto knots = openKnots( 7 );
            std::vector<api::PatchGeometry> patches;
            for( size_t patch = 0; patch < 2; patch++ )
                patches.push_back( { basis::KnotVector( knots, 1e-10 ), basis::KnotVector( knots, 1e-10 ), 3, 3,
                                     controlPoints( knots, 7.0 * patch, false ) } );
            const std::vector<std::pair<api::PatchSideRef, api::PatchSideRef>> gluing{
                { { 0, api::PatchSide::S1 }, { 1, api::PatchSide::S0 } } };
            std::vector<std::vector<api::PatchElement>> marks( depth );
            for( size_t level = 0; level < depth; level++ )
                for( size_t j = 0; j < 4; j++ )
                    for( size_t i = 0; i < 4; i++ )
                    {
                        marks.at( level ).push_back( { 0, ( 7UL << level ) - 4 + i, j } );
                        marks.at( level ).push_back( { 1, i, j } );
                    }
            measure( fixture, depth, 98 + 96 * depth, [&] {
                return std::make_unique<api::NavierStokesHierarchicalMultiPatchDiscretization>(
                    patches, gluing, marks );
            } );
            return;
        }
        const size_t n = fixture == "F1n" ? 6 : 7;
        const auto knots = openKnots( n );
        std::vector<std::vector<std::pair<size_t, size_t>>> marks( depth );
        size_t leaves = n * n;
        for( size_t level = 0; level < depth; level++ )
        {
            const size_t width = fixture == "F1w" ? ( level == 0 ? 6 : level == 1 ? 10 : 16 ) : 4;
            for( size_t j = 0; j < width; j++ )
                for( size_t i = 0; i < width; i++ ) marks.at( level ).emplace_back( i, j );
            leaves += 3 * width * width;
        }
        measure( fixture, depth, leaves, [&]() -> std::unique_ptr<api::NavierStokesHierarchicalDiscretization> {
            if( fixture == "F1n" )
                return std::make_unique<api::NURBSNavierStokesHierarchicalDiscretization>(
                    basis::KnotVector( knots, 1e-10 ), basis::KnotVector( knots, 1e-10 ), 3, 3,
                    controlPoints( knots, 0.0, true ), marks );
            return std::make_unique<api::NavierStokesHierarchicalDiscretization>(
                basis::KnotVector( knots, 1e-10 ), basis::KnotVector( knots, 1e-10 ), 3, 3,
                controlPoints( knots, 0.0, false ), marks );
        } );
    }
}

int main( const int argc, char** argv )
{
    const std::string usage = "usage: thb_activity_benchmark F1|F1n|F1w|F2 depth(0..7)";
    if( argc == 2 and std::string( argv[1] ) == "--help" ) { std::cout << usage << '\n'; return 0; }
    try
    {
        if( argc != 3 ) throw std::invalid_argument( usage );
        const std::string fixture = argv[1], depth_text = argv[2];
        if( fixture != "F1" and fixture != "F1n" and fixture != "F1w" and fixture != "F2" )
            throw std::invalid_argument( "Unknown fixture; " + usage );
        if( depth_text.size() != 1 or depth_text.front() < '0' or depth_text.front() > '7' )
            throw std::invalid_argument( "Invalid depth; " + usage );
        run( fixture, static_cast<size_t>( depth_text.front() - '0' ) );
        return 0;
    }
    catch( const std::exception& e )
    {
        std::cerr << "THB benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
