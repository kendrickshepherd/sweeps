#include <catch2/catch_test_macros.hpp>
#include "support/ThbActivityChecks.hpp"
#include "support/ThbActivityFixtures.hpp"
#include <sstream>

using namespace basis;
using namespace topology;
using namespace thb_test;

namespace
{
    // Independent single-patch support oracle: knot intervals and requested index sets only.
    // Does not use connectivity(), iterateChildren(), or either activity implementation.
    std::set<FunctionId> supportOracle( const TPSplineSpace& space, const FixtureLevels& fixture, const size_t level )
    {
        const auto components = tensorProductComponentSplines( space );
        Index cells, functions, degrees;
        for( const auto& component : components )
        {
            const size_t degree = component->basisComplex().defaultParentBasis().mBasisGroups.at( 0 ).degrees.at( 0 );
            cells.push_back( numElements( component->knotVector() ) );
            functions.push_back( component->knotVector().size() - degree - 1 );
            degrees.push_back( degree );
        }
        std::set<FunctionId> expected;
        size_t fid = 0;
        eachIndex( functions, [&]( const Index& function ) {
            bool contained = true, touches_leaf = false, nonempty = false;
            eachIndex( cells, [&]( const Index& cell ) {
                for( size_t axis = 0; axis < cells.size(); axis++ )
                {
                    const auto& kv = components.at( axis )->knotVector();
                    const auto breaks = kv.uniqueKnots();
                    if( kv.knot( function.at( axis ) ) > breaks.at( cell.at( axis ) ) + 1e-12 or
                        breaks.at( cell.at( axis ) + 1 ) >
                            kv.knot( function.at( axis ) + degrees.at( axis ) + 1 ) + 1e-12 ) return;
                }
                nonempty = true;
                Index ancestor = cell;
                for( size_t coarse = level; coarse > 0; )
                {
                    --coarse;
                    const auto& ratio = fixture.ratios.at( coarse ).front();
                    for( size_t axis = 0; axis < cells.size(); axis++ ) ancestor.at( axis ) /= ratio.at( axis );
                    if( fixture.leaves.at( coarse ).front().contains( ancestor ) ) contained = false;
                }
                touches_leaf = touches_leaf or fixture.leaves.at( level ).front().contains( cell );
            } );
            if( not nonempty ) throw std::runtime_error( "Oracle function has empty support" );
            if( contained and touches_leaf ) expected.insert( FunctionId( fid ) );
            fid++;
        } );
        return expected;
    }

    void checkSupportOracle( const HierarchicalTPSplineSpace& ss, const FixtureLevels& fixture )
    {
        for( size_t level = 0; level < ss.refinementLevels().size(); level++ )
        {
            const auto& stored = ss.activeFunctions().at( level );
            requireSameActiveFunctions( supportOracle( *ss.refinementLevels().at( level ), fixture, level ),
                                        std::set<FunctionId>( stored.begin(), stored.end() ),
                                        "knot-support oracle level " + std::to_string( level ) );
        }
    }

    void checkSingle( const FixtureSpec& spec )
    {
        const auto fixture = fixtureLevels( spec );
        std::vector<std::shared_ptr<const TPSplineSpace>> levels, l2_levels;
        for( const auto& patches : fixture.patches )
        {
            levels.push_back( patches.front() );
            SmallVector<KnotVector, 3> knots;
            SmallVector<size_t, 3> degrees;
            for( const auto& component : tensorProductComponentSplines( *patches.front() ) )
            {
                knots.push_back( reducedOrder( component->knotVector() ) );
                degrees.push_back( spec.degree - 1 );
            }
            l2_levels.push_back( std::make_shared<const TPSplineSpace>( buildBSpline( knots, degrees ) ) );
        }
        const auto leaves = tensorLeaves( fixture );
        const auto h1 = buildHierarchicalSplineSpace( levels, leaves );
        std::vector<std::set<Index>> actual_leaves( levels.size() );
        for( const auto& info : leafElementInfos( h1.basisComplex().parametricAtlas().cmap() ) )
            actual_leaves.at( info.level ).insert( Index( info.indices.begin(), info.indices.end() ) );
        for( size_t level = 0; level < levels.size(); level++ )
            if( actual_leaves.at( level ) != fixture.leaves.at( level ).front() )
                throw std::runtime_error( "Constructed leaves differ from requested leaves" );
        const auto& actual_ratios = h1.basisComplex().parametricAtlas().cmap().refinementRatios();
        for( size_t level = 0; level < spec.ratios.size(); level++ )
            for( size_t axis = 0; axis < spec.dim; axis++ )
                if( actual_ratios.at( level ).at( axis ) != spec.ratios.at( level ).at( axis ) )
                    throw std::runtime_error( "Fixture refinement ratio changed" );
        const auto hdiv_bc = std::make_shared<const VectorConformingBasisComplex>(
            h1.basisComplexPtr(), ConformingType::Divergence );
        const auto hcurl_bc = std::make_shared<const VectorConformingBasisComplex>(
            h1.basisComplexPtr(), ConformingType::Curl );
        const VectorConformingHierarchicalTPSplineSpace hdiv( hdiv_bc, h1 ), hcurl( hcurl_bc, h1 );
        const auto l2 = buildHierarchicalSplineSpace( l2_levels, leaves );
        ActivityReport report;
        checkActivity( h1, "H1", report );
        checkActivity( hdiv, "Hdiv", report );
        checkActivity( hcurl, "Hcurl", report );
        checkActivity( l2, "L2", report );
        CHECK( report.levels == ( 2 + 2 * spec.dim ) * levels.size() );
        checkSupportOracle( h1, fixture );
        checkSupportOracle( l2, fixture );
        for( const auto* vector : { &hdiv, &hcurl } )
            for( const auto& scalar : vector->scalarBases() ) checkSupportOracle( *scalar, fixture );
    }

    void checkMulti( const FixtureSpec& spec )
    {
        const auto fixture = fixtureLevels( spec );
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        for( const auto& patches : fixture.patches )
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, fixture.connections ) ) );
        const auto h1 = buildHierarchicalSplineSpace( levels, multiLeaves( fixture, levels ) );
        LeafIndices actual( fixture.leaves.size(), std::vector<std::set<Index>>( spec.patches ) );
        for( const auto& info : leafElementInfos( h1.basisComplex().parametricAtlas().cmap() ) )
            actual.at( info.level ).at( info.patch ).insert( Index( info.indices.begin(), info.indices.end() ) );
        if( actual != fixture.leaves ) throw std::runtime_error( "Constructed multipatch leaves differ from request" );
        if( spec.marks == "random" and spec.patches == 2 )
        {
            bool asymmetric_counts = false;
            for( const auto& level : fixture.leaves )
                asymmetric_counts = asymmetric_counts or level.front().size() != level.back().size();
            if( not asymmetric_counts ) throw std::runtime_error( "Fixture failed to produce asymmetric leaf counts" );
        }
        const auto hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const auto hcurl = buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        const auto l2 = buildL2HierarchicalMultiPatchSplineSpace( hdiv );
        ActivityReport report;
        checkActivity( h1, "H1 global", report );
        checkActivity( hdiv, "Hdiv global", report );
        checkActivity( hcurl, "Hcurl global", report );
        checkActivity( l2, "L2 global", report );
        CHECK( report.levels == 4 * levels.size() );
    }

    FixtureSpec specFor( const size_t dim )
    {
        FixtureSpec spec;
        spec.dim = dim;
        spec.cells = dim == 2 ? Index{ 2, 2 } : Index{ 2, 1, 2 };
        spec.ratios = dim == 2 ? std::vector<Index>{ { 2, 1 }, { 1, 3 } }
                              : std::vector<Index>{ { 2, 1, 1 }, { 1, 3, 1 } };
        spec.permutation = dim == 2 ? Permutation::Flip1d : Permutation::ZeroToZero;
        return spec;
    }
}

TEST_CASE( "Activity reference and knot supports agree on tensor-product hierarchies", "[thb_reference]" )
{
    for( const size_t dim : { 2, 3 } )
        for( const size_t degree : { 1, 2, 3 } )
        {
            auto spec = specFor( dim );
            spec.degree = degree;
            spec.ratios.clear();
            CAPTURE( dim, degree );
            CHECK_NOTHROW( checkSingle( spec ) );
        }
    for( const size_t dim : { 2, 3 } )
        for( const size_t degree : { 1, 2, 3 } )
            for( const std::string marks : { "none", "uniform", "random" } )
            {
                auto spec = specFor( dim );
                spec.degree = degree;
                spec.marks = marks;
                CAPTURE( dim, degree, marks );
                CHECK_NOTHROW( checkSingle( spec ) );
            }
}

TEST_CASE( "Activity reference covers fixed-seed histories and C0 reduced spaces", "[thb_reference]" )
{
    for( const uint32_t seed : { 17U, 1729U, 8675309U } )
        for( const size_t dim : { 2, 3 } )
        {
            auto spec = specFor( dim );
            spec.seed = seed;
            spec.c0 = true;
            CAPTURE( seed, dim );
            CHECK_NOTHROW( checkSingle( spec ) );
        }
}

TEST_CASE( "Activity reference preserves the full-support level-jump fix", "[thb_reference]" )
{
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.cells = dim == 2 ? Index{ 2, 1 } : Index{ 2, 1, 1 };
        spec.ratios.assign( 2, Index( dim, 1 ) );
        for( auto& ratio : spec.ratios ) ratio.front() = 2;
        spec.marks = "jump";
        CAPTURE( dim );
        CHECK_NOTHROW( checkSingle( spec ) );
        const auto fixture = fixtureLevels( spec );
        std::vector<std::shared_ptr<const TPSplineSpace>> levels;
        for( const auto& patch : fixture.patches ) levels.push_back( patch.front() );
        const auto ss = buildHierarchicalSplineSpace( levels, tensorLeaves( fixture ) );
        // For p=2 the first-direction id 3 reaches outside Omega^1, for every tangential id.
        const auto& active = ss.activeFunctions().at( 1 );
        CHECK( active.size() == ( dim == 2 ? 6 : 18 ) );
        for( size_t fid = 3; fid < levels.at( 1 )->numFunctions(); fid += 6 )
            CHECK_FALSE( std::set<FunctionId>( active.begin(), active.end() ).contains( FunctionId( fid ) ) );
    }
}

TEST_CASE( "Activity reference covers asymmetric and reoriented multipatch hierarchies", "[thb_reference]" )
{
    bool saw_reversed = false, saw_swapped = false;
    for( const size_t dim : { 2, 3 } )
        for( const bool reverse : { false, true } )
            for( const uint32_t seed : { 17U, 1729U } )
            {
                auto spec = specFor( dim );
                spec.patches = 2;
                spec.reverse = reverse;
                spec.seed = seed;
                if( dim == 3 and reverse ) spec.permutation = Permutation::ZeroToOne;
                const auto transform = sideCoordinateTransform( dim, 0, reverse ? 0 : 1, spec.permutation );
                for( size_t d = 0; d + 1 < dim; d++ )
                {
                    saw_reversed = saw_reversed or transform.source_axis_reversed.at( d );
                    saw_swapped = saw_swapped or transform.source_axis_for_destination.at( d ) != d;
                }
                CAPTURE( dim, reverse, seed );
                CHECK_NOTHROW( checkMulti( spec ) );
            }
    CHECK( saw_reversed );
    CHECK( saw_swapped );
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.patches = 2;
        spec.cells.assign( dim, 1 );
        spec.cells.front() = 2;
        spec.ratios.assign( 2, Index( dim, 1 ) );
        for( auto& ratio : spec.ratios ) ratio.front() = 2;
        spec.marks = "jump";
        CAPTURE( dim );
        CHECK_NOTHROW( checkMulti( spec ) );
    }
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.patches = 4;
        spec.cells.assign( dim, 1 );
        spec.ratios = { Index( dim, 2 ) };
        spec.marks = "diagonal";
        CAPTURE( dim );
        CHECK_NOTHROW( checkMulti( spec ) );
    }
}

TEST_CASE( "Activity comparison fails on equal-sized wrong sets and constructor-only defects", "[thb_reference]" )
{
    const std::set<FunctionId> expected{ FunctionId( 1 ), FunctionId( 4 ) };
    CHECK_NOTHROW( requireSameActiveFunctions( expected, expected, "control" ) );
    try
    {
        requireSameActiveFunctions( expected, { FunctionId( 1 ), FunctionId( 5 ) }, "equal-sized control" );
        FAIL_CHECK( "An equal-sized wrong set was accepted" );
    }
    catch( const std::runtime_error& e )
    {
        CHECK( std::string( e.what() ).find( "missing 4; extra 5" ) != std::string::npos );
    }
    auto spec = specFor( 2 );
    spec.marks = "uniform";
    const auto fixture = fixtureLevels( spec );
    std::vector<std::shared_ptr<const TPSplineSpace>> levels;
    for( const auto& patch : fixture.patches ) levels.push_back( patch.front() );
    const auto good = buildHierarchicalSplineSpace( levels, tensorLeaves( fixture ) );
    auto wrong = good.activeFunctions();
    REQUIRE_FALSE( wrong.back().empty() );
    wrong.back().pop_back();
    const HierarchicalTPSplineSpace bad( good.basisComplexPtr(), levels, wrong );
    ActivityReport report;
    try
    {
        checkActivity( bad, "constructor-only control", report );
        FAIL_CHECK( "A constructor-only mismatch was accepted" );
    }
    catch( const std::runtime_error& e )
    {
        CHECK( std::string( e.what() ).find( "constructor-only control level 2 constructor" ) != std::string::npos );
    }
}
