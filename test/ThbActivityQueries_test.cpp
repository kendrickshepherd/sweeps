#include <catch2/catch_test_macros.hpp>
#include "support/ThbActivityChecks.hpp"
#include "support/ThbActivityFixtures.hpp"
#include <ThbLevelDomains.hpp>
#include <ThbLocalSupport.hpp>
#include <ParametricAtlas1d.hpp>
#include <array>
#include <map>
#include <optional>
#include <tuple>

using namespace basis;
using namespace topology;
using namespace thb_test;
using basis::detail::SupportRanges1d;
using basis::detail::TensorSupport;
using topology::detail::LevelDomains;

namespace
{
    using Range = std::pair<size_t, size_t>;
    using PatchCell = std::pair<size_t, Index>;
    using SpaceList = std::vector<std::shared_ptr<const TPSplineSpace>>;

    LevelDomains::Indices small( const Index& ix ) { return LevelDomains::Indices( ix.begin(), ix.end() ); }

    std::shared_ptr<const BSplineSpace1d> bspline1d( const std::vector<double>& knots, const size_t degree,
                                                     const std::optional<size_t> cmap_elements = std::nullopt )
    {
        const KnotVector kv( knots, 1e-12 );
        const size_t elements = cmap_elements.value_or( numElements( kv ) );
        const auto cmap = std::make_shared<const CombinatorialMap1d>( elements );
        const Eigen::VectorXd lengths = parametricLengths( kv ).head( elements );
        const auto atlas = std::make_shared<const param::ParametricAtlas1d>( cmap, lengths );
        return std::make_shared<const BSplineSpace1d>( std::make_shared<const BasisComplex1d>( atlas, degree ), kv );
    }

    std::vector<Range> knotRanges( const BSplineSpace1d& space )
    {
        const SupportRanges1d ranges( space );
        std::vector<Range> out;
        for( size_t i = 0; i < ranges.numFunctions(); i++ ) out.push_back( ranges.range( i ) );
        return out;
    }

    // Oracle: the elements whose connectivity lists each function.
    void checkRanges1d( const BSplineSpace1d& space, const std::string& where )
    {
        const auto& cmap = space.basisComplex().parametricAtlas().cmap();
        std::vector<std::set<size_t>> expected( space.numFunctions() );
        iterateCellsWhile( cmap, 1, [&]( const Cell& e ) {
            for( const FunctionId& fid : space.connectivity( e ) ) expected.at( fid.id() ).insert( e.dart().id() );
            return true;
        } );
        const SupportRanges1d ranges( space );
        if( ranges.numFunctions() != expected.size() or ranges.numElements() != cellCount( cmap, 1 ) )
            throw std::runtime_error( where + ": 1d sizes differ" );
        for( size_t i = 0; i < expected.size(); i++ )
        {
            std::set<size_t> actual;
            for( size_t e = ranges.range( i ).first; e < ranges.range( i ).second; e++ ) actual.insert( e );
            if( expected.at( i ).empty() or actual != expected.at( i ) )
                throw std::runtime_error( where + ": 1d function " + std::to_string( i ) );
        }
    }

    // Oracle: tensorCellSupports, i.e. connectivity inverted over every cell.
    size_t checkTensorSupport( const TPSplineSpace& space, TensorSupport::Cache& cache, const std::string& where )
    {
        const TensorSupport support( space, cache );
        const auto expected = tensorCellSupports( space );
        if( support.numFunctions() != space.numFunctions() or expected.size() != space.numFunctions() )
            throw std::runtime_error( where + ": tensor sizes differ" );
        for( size_t fid = 0; fid < expected.size(); fid++ )
        {
            std::vector<Index> cells;
            const bool finished = support.iterateCellsWhile( FunctionId( fid ), [&]( const TensorSupport::Indices& c ) {
                cells.emplace_back( c.begin(), c.end() );
                return true;
            } );
            size_t product = 1;
            for( const auto& [first, last] : support.ranges( FunctionId( fid ) ) ) product *= last - first;
            std::set<Index> actual( cells.begin(), cells.end() ), wanted;
            for( const auto& c : expected.at( fid ) ) wanted.emplace( c.begin(), c.end() );
            if( not finished or cells.size() != product or actual.size() != cells.size() or
                wanted.size() != expected.at( fid ).size() or actual != wanted )
                throw std::runtime_error( where + ": tensor function " + std::to_string( fid ) );
        }
        return expected.size();
    }

    struct Counts
    {
        size_t spaces = 0;
        size_t functions = 0;
        size_t outside = 0;
        size_t deep = 0;
    };

    void checkSpaces( const SpaceList& spaces, const std::string& where, Counts& counts )
    {
        TensorSupport::Cache cache;
        for( size_t i = 0; i < spaces.size(); i++ )
        {
            const std::string name = where + " space " + std::to_string( i );
            for( const auto& component : tensorProductComponentSplines( *spaces.at( i ) ) )
                checkRanges1d( *component, name );
            counts.functions += checkTensorSupport( *spaces.at( i ), cache, name );
            counts.spaces++;
        }
    }

    Index unflatIndex( const TPCombinatorialMap& cmap, const Dart& d )
    {
        Index out;
        for( const Dart& unflat : unflattenFull( cmap, d ).unflat_darts ) out.push_back( unflat.id() );
        return out;
    }

    PatchCell locate( const HierarchicalTPCombinatorialMap& cmap, const size_t level, const Cell& c )
    {
        return { 0, unflatIndex( *cmap.refinementLevels().at( level ), c.dart() ) };
    }

    PatchCell locate( const HierarchicalMultiPatchCombinatorialMap& cmap, const size_t level, const Cell& c )
    {
        const MultiPatchCombinatorialMap& level_cmap = *cmap.refinementLevels().at( level );
        const auto [patch, local] = level_cmap.toLocalDart( c.dart() );
        return { patch, unflatIndex( *level_cmap.constituents().at( patch ), local ) };
    }

    template <typename CMap>
    std::vector<Cell> childrenOf( const CMap& cmap, const std::vector<Cell>& cells, const size_t level )
    {
        std::vector<Cell> out;
        for( const Cell& c : cells )
            cmap.iterateChildren( c, level, [&]( const Cell& child ) {
                out.push_back( child );
                return true;
            } );
        return out;
    }

    // Oracle: the cells covered by a coarser leaf through iterateChildren, as levelActiveFunctions builds them.
    template <typename CMap>
    void checkRegion( const CMap& cmap, const FixtureLevels& fixture, const std::string& where, Counts& counts )
    {
        const auto leaves = leafElements( cmap );
        const LevelDomains domains = topology::detail::levelDomains( cmap, leaves );
        const size_t num_patches = fixture.patches.front().size();
        if( domains.numPatches() != num_patches ) throw std::runtime_error( where + ": patch count" );
        std::vector<Cell> covered;
        for( size_t level = 0; level < leaves.size(); level++ )
        {
            std::set<PatchCell> all, from_parent;
            for( const Cell& c : covered ) all.insert( locate( cmap, level, c ) );
            if( level > 0 )
                for( const Cell& c : childrenOf( cmap, leaves.at( level - 1 ), level - 1 ) )
                    from_parent.insert( locate( cmap, level, c ) );
            for( size_t patch = 0; patch < num_patches; patch++ )
            {
                const std::string at =
                    where + " level " + std::to_string( level ) + " patch " + std::to_string( patch );
                if( domains.numLevels( patch ) != leaves.size() ) throw std::runtime_error( at + ": level count" );
                Index lengths;
                for( const auto& component : tensorProductComponentSplines( *fixture.patches.at( level ).at( patch ) ) )
                    lengths.push_back( numElements( component->knotVector() ) );
                const auto& stored = domains.lengths( patch, level );
                if( Index( stored.begin(), stored.end() ) != lengths ) throw std::runtime_error( at + ": lengths" );
                eachIndex( lengths, [&]( const Index& ix ) {
                    const bool inside = domains.contains( patch, level, small( ix ) );
                    if( inside == all.contains( { patch, ix } ) ) throw std::runtime_error( at + ": membership" );
                    counts.outside += not inside;
                    counts.deep += not inside and not from_parent.contains( { patch, ix } );
                } );
            }
            covered.insert( covered.end(), leaves.at( level ).begin(), leaves.at( level ).end() );
            if( level + 1 < leaves.size() ) covered = childrenOf( cmap, covered, level );
        }
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

    template <typename Levels>
    void appendScalarLevels( const Levels& scalar_bases, SpaceList& spaces )
    {
        for( const auto& scalar : scalar_bases )
            spaces.insert( spaces.end(), scalar->refinementLevels().begin(), scalar->refinementLevels().end() );
    }

    Counts checkSingle( const FixtureSpec& spec )
    {
        const auto fixture = fixtureLevels( spec );
        SpaceList levels, l2_levels;
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
        const auto h1 = buildHierarchicalSplineSpace( levels, tensorLeaves( fixture ) );
        const auto hdiv_bc = std::make_shared<const VectorConformingBasisComplex>(
            h1.basisComplexPtr(), ConformingType::Divergence );
        const auto hcurl_bc = std::make_shared<const VectorConformingBasisComplex>(
            h1.basisComplexPtr(), ConformingType::Curl );
        const VectorConformingHierarchicalTPSplineSpace hdiv( hdiv_bc, h1 ), hcurl( hcurl_bc, h1 );
        SpaceList spaces = levels;
        spaces.insert( spaces.end(), l2_levels.begin(), l2_levels.end() );
        appendScalarLevels( hdiv.scalarBases(), spaces );
        appendScalarLevels( hcurl.scalarBases(), spaces );
        Counts counts;
        checkSpaces( spaces, "single", counts );
        checkRegion( h1.basisComplex().parametricAtlas().cmap(), fixture, "single", counts );
        return counts;
    }

    template <typename LevelList>
    void appendPatchSpaces( const LevelList& levels, SpaceList& spaces )
    {
        for( const auto& level : levels )
            spaces.insert( spaces.end(), level->subSpaces().begin(), level->subSpaces().end() );
    }

    template <typename LevelList>
    void appendVectorPatchSpaces( const LevelList& levels, SpaceList& spaces )
    {
        for( const auto& level : levels )
            for( const auto& patch : level->subSpaces() )
                spaces.insert( spaces.end(), patch->scalarTPBases().begin(), patch->scalarTPBases().end() );
    }

    Counts checkMultiSpaces( const FixtureLevels& fixture )
    {
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        for( const auto& patches : fixture.patches )
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, fixture.connections ) ) );
        const auto h1 = buildHierarchicalSplineSpace( levels, multiLeaves( fixture, levels ) );
        const auto hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const auto hcurl = buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        const auto l2 = buildL2HierarchicalMultiPatchSplineSpace( hdiv );
        SpaceList spaces;
        appendPatchSpaces( h1.refinementLevels(), spaces );
        appendPatchSpaces( l2.refinementLevels(), spaces );
        appendVectorPatchSpaces( hdiv.refinementLevels(), spaces );
        appendVectorPatchSpaces( hcurl.refinementLevels(), spaces );
        Counts counts;
        checkSpaces( spaces, "multi", counts );
        checkRegion( h1.basisComplex().parametricAtlas().cmap(), fixture, "multi", counts );
        return counts;
    }

    void add( Counts& total, const Counts& more )
    {
        total.spaces += more.spaces;
        total.functions += more.functions;
        total.outside += more.outside;
        total.deep += more.deep;
    }

    template <typename Callable>
    void requireThrows( Callable&& call, const std::string& type )
    {
        try
        {
            call();
        }
        catch( const std::invalid_argument& )
        {
            CHECK( type == "invalid_argument" );
            return;
        }
        catch( const std::out_of_range& )
        {
            CHECK( type == "out_of_range" );
            return;
        }
        catch( const std::logic_error& )
        {
            CHECK( type == "logic_error" );
            return;
        }
        FAIL_CHECK( "Expected " + type );
    }

    // A non-B-spline source with the same basis complex as a B-spline one.
    class WrappedSpace final : public SplineSpace
    {
        public:
        explicit WrappedSpace( const std::shared_ptr<const SplineSpace>& inner ) : mInner( inner ) {}
        const BasisComplex& basisComplex() const override { return mInner->basisComplex(); }
        Eigen::MatrixXd extractionOperator( const Cell& c ) const override { return mInner->extractionOperator( c ); }
        std::vector<FunctionId> connectivity( const Cell& c ) const override { return mInner->connectivity( c ); }
        size_t numFunctions() const override { return mInner->numFunctions(); }

        private:
        const std::shared_ptr<const SplineSpace> mInner;
    };
}

TEST_CASE( "1d support ranges match hand examples and inverted connectivity", "[thb_queries]" )
{
    CHECK( knotRanges( *bspline1d( { 0, 0, 0, 1, 2, 2, 3, 3, 3 }, 2 ) ) ==
           std::vector<Range>{ { 0, 1 }, { 0, 2 }, { 0, 2 }, { 1, 3 }, { 2, 3 }, { 2, 3 } } );
    CHECK( knotRanges( *bspline1d( { 0, 1, 2, 3 }, 0 ) ) == std::vector<Range>{ { 0, 1 }, { 1, 2 }, { 2, 3 } } );
    CHECK( knotRanges( *bspline1d( { 0, 0, 1, 1, 2, 2 }, 1 ) ) ==
           std::vector<Range>{ { 0, 1 }, { 0, 1 }, { 1, 2 }, { 1, 2 } } );

    size_t cases = 0;
    for( size_t degree = 0; degree <= 3; degree++ )
        for( size_t multiplicity = 1; multiplicity <= degree + 1; multiplicity++ )
        {
            std::vector<double> knots( degree + 1, 0.0 );
            knots.push_back( 1.0 );
            knots.insert( knots.end(), multiplicity, 2.5 );
            knots.insert( knots.end(), std::min<size_t>( 2, degree + 1 ), 4.0 );
            knots.insert( knots.end(), degree + 1, 7.0 );
            CAPTURE( degree, multiplicity );
            CHECK_NOTHROW( checkRanges1d( *bspline1d( knots, degree ), "1d" ) );
            cases++;
        }
    CHECK( cases == 10 );
}

TEST_CASE( "Tensor supports and level domains match their oracles on single-patch hierarchies", "[thb_queries]" )
{
    Counts total;
    for( const size_t dim : { 2, 3 } )
        for( const size_t degree : { 1, 2, 3 } )
            for( const std::string marks : { "none", "uniform", "random" } )
                for( const bool c0 : { false, true } )
                {
                    auto spec = specFor( dim );
                    spec.degree = degree;
                    spec.marks = marks;
                    spec.c0 = c0;
                    CAPTURE( dim, degree, marks, c0 );
                    Counts counts;
                    CHECK_NOTHROW( counts = checkSingle( spec ) );
                    add( total, counts );
                }
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.cells = dim == 2 ? Index{ 2, 1 } : Index{ 2, 1, 1 };
        spec.ratios.assign( 2, Index( dim, 1 ) );
        for( auto& ratio : spec.ratios ) ratio.front() = 2;
        spec.marks = "jump";
        CAPTURE( dim );
        Counts counts;
        CHECK_NOTHROW( counts = checkSingle( spec ) );
        add( total, counts );
    }
    // Square levels with axis-0 marks: only membership can witness the axis order of leaf indices.
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.cells.assign( dim, 2 );
        spec.ratios.assign( 2, Index( dim, 2 ) );
        spec.marks = "jump";
        CAPTURE( dim );
        Counts counts;
        CHECK_NOTHROW( counts = checkSingle( spec ) );
        CHECK( counts.deep > 0 );
        add( total, counts );
    }
    CHECK( total.spaces > 0 );
    CHECK( total.functions > 0 );
    CHECK( total.outside > 0 );
    CHECK( total.deep > 0 );
}

TEST_CASE( "Tensor supports and level domains match their oracles on multipatch hierarchies", "[thb_queries]" )
{
    Counts total;
    for( const size_t dim : { 2, 3 } )
        for( const bool reverse : { false, true } )
            for( const uint32_t seed : { 17U, 1729U } )
            {
                auto spec = specFor( dim );
                spec.patches = 2;
                spec.reverse = reverse;
                spec.seed = seed;
                if( dim == 3 and reverse ) spec.permutation = Permutation::ZeroToOne;
                CAPTURE( dim, reverse, seed );
                Counts counts;
                CHECK_NOTHROW( counts = checkMultiSpaces( fixtureLevels( spec ) ) );
                add( total, counts );
            }
    for( const size_t dim : { 2, 3 } )
    {
        auto spec = specFor( dim );
        spec.patches = 4;
        spec.cells.assign( dim, 1 );
        spec.ratios = { Index( dim, 2 ) };
        spec.marks = "diagonal";
        CAPTURE( dim );
        Counts counts;
        CHECK_NOTHROW( counts = checkMultiSpaces( fixtureLevels( spec ) ) );
        add( total, counts );
    }
    CHECK( total.spaces > 0 );
    CHECK( total.functions > 0 );
    CHECK( total.outside > 0 );
    CHECK( total.deep > 0 );
}

TEST_CASE( "Tensor support iteration is axis 0 fastest and stops on request", "[thb_queries]" )
{
    const TPSplineSpace space = buildBSpline( { KnotVector( { 0, 0, 0, 1, 2, 3, 3, 3 }, 1e-12 ),
                                                KnotVector( { 0, 0, 0, 1, 2, 3, 4, 4, 4 }, 1e-12 ) },
                                              { 2, 2 } );
    TensorSupport::Cache cache;
    const TensorSupport support( space, cache );
    CHECK( cache.size() == 2 );
    const TensorSupport again( space, cache );
    CHECK( cache.size() == 2 );

    // Function (1, 2): ranges [0, 2) x [0, 3).
    const FunctionId fid( 1 + 5 * 2 );
    REQUIRE( support.ranges( fid ) == TensorSupport::Ranges{ { 0, 2 }, { 0, 3 } } );
    std::vector<Index> all;
    CHECK( support.iterateCellsWhile( fid, [&]( const TensorSupport::Indices& c ) {
        all.emplace_back( c.begin(), c.end() );
        return true;
    } ) );
    CHECK( all == std::vector<Index>{ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 0, 2 }, { 1, 2 } } );
    for( size_t stop = 1; stop <= all.size(); stop++ )
    {
        std::vector<Index> seen;
        const bool finished = support.iterateCellsWhile( fid, [&]( const TensorSupport::Indices& c ) {
            seen.emplace_back( c.begin(), c.end() );
            return seen.size() < stop;
        } );
        CAPTURE( stop );
        CHECK_FALSE( finished );
        CHECK( seen == std::vector<Index>( all.begin(), all.begin() + stop ) );
    }
}

TEST_CASE( "Level domains match a hand example over two transitions", "[thb_queries]" )
{
    const std::vector<LevelDomains::PatchLevels> patches{ { { 2, 2 }, { { 2, 1 }, { 1, 3 } } },
                                                          { { 1, 1 }, { { 1, 1 }, { 1, 1 } } } };
    // Unsorted on purpose: LevelDomains must order the leaves itself.
    std::vector<LevelDomains::Leaf> leaves{ { 0, 0, { 0, 0 } }, { 0, 1, { 2, 0 } }, { 0, 1, { 2, 1 } },
                                            { 0, 1, { 0, 1 } }, { 0, 1, { 1, 1 } }, { 1, 0, { 0, 0 } } };
    for( size_t j = 0; j < 6; j++ ) leaves.push_back( { 0, 2, { 3, j } } );
    const LevelDomains domains( patches, leaves );
    CHECK( domains.numPatches() == 2 );
    CHECK( domains.numLevels( 0 ) == 3 );
    CHECK( domains.lengths( 0, 2 ) == LevelDomains::Indices{ 4, 6 } );
    CHECK( domains.lengths( 1, 2 ) == LevelDomains::Indices{ 1, 1 } );

    const std::vector<std::tuple<size_t, size_t, Index, bool>> expected{
        { 0, 0, { 0, 0 }, true },  { 0, 1, { 0, 0 }, false }, { 0, 1, { 1, 0 }, false }, { 0, 1, { 2, 0 }, true },
        { 0, 2, { 0, 0 }, false }, { 0, 2, { 0, 3 }, false }, { 0, 2, { 2, 0 }, false }, { 0, 2, { 2, 5 }, false },
        { 0, 2, { 1, 5 }, false }, { 0, 2, { 3, 4 }, true },  { 0, 2, { 1, 1 }, false }, { 1, 0, { 0, 0 }, true },
        { 1, 1, { 0, 0 }, false }, { 1, 2, { 0, 0 }, false } };
    for( const auto& [patch, level, cell, inside] : expected )
    {
        CAPTURE( patch, level, cell );
        CHECK( domains.contains( patch, level, small( cell ) ) == inside );
    }
}

TEST_CASE( "Query guards throw their documented types", "[thb_queries]" )
{
    using PL = LevelDomains::PatchLevels;
    const PL good{ { 2, 2 }, { { 2, 1 } } };
    requireThrows( [] { LevelDomains( {}, {} ); }, "invalid_argument" );
    requireThrows( [] { LevelDomains( { PL{ {}, {} } }, {} ); }, "invalid_argument" );
    requireThrows( [&] { LevelDomains( { good, PL{ { 1, 1, 1 }, { { 1, 1, 1 } } } }, {} ); }, "invalid_argument" );
    requireThrows( [] { LevelDomains( { PL{ { 2, 2 }, { { 2, 1, 1 } } } }, {} ); }, "invalid_argument" );
    requireThrows( [] { LevelDomains( { PL{ { 2, 0 }, {} } }, {} ); }, "invalid_argument" );
    requireThrows( [] { LevelDomains( { PL{ { 2, 2 }, { { 0, 1 } } } }, {} ); }, "invalid_argument" );
    for( const LevelDomains::Leaf& leaf : std::vector<LevelDomains::Leaf>{
             { 1, 0, { 0, 0 } }, { 0, 2, { 0, 0 } }, { 0, 1, { 4, 0 } }, { 0, 0, { 0, 2 } }, { 0, 0, { 0 } } } )
    {
        CAPTURE( leaf.patch, leaf.level, leaf.indices.size() );
        requireThrows( [&] { LevelDomains( { good }, { leaf } ); }, "invalid_argument" );
    }

    const LevelDomains domains( { good }, { { 0, 0, { 0, 0 } } } );
    CHECK_NOTHROW( domains.contains( 0, 1, { 3, 1 } ) );
    for( const auto& [patch, level, cell] : std::vector<std::tuple<size_t, size_t, LevelDomains::Indices>>{
             { 1, 0, { 0, 0 } }, { 0, 2, { 0, 0 } }, { 0, 1, { 4, 0 } }, { 0, 1, { 0, 2 } }, { 0, 0, { 0 } },
             { 0, 0, { 0, 0, 0 } } } )
    {
        CAPTURE( patch, level, cell.size() );
        requireThrows( [&] { domains.contains( patch, level, cell ); }, "out_of_range" );
    }

    auto spec = specFor( 2 );
    const auto fixture = fixtureLevels( spec );
    SpaceList levels;
    for( const auto& patch : fixture.patches ) levels.push_back( patch.front() );
    const auto h1 = buildHierarchicalSplineSpace( levels, tensorLeaves( fixture ) );
    const auto& tp_cmap = h1.basisComplex().parametricAtlas().cmap();
    auto tp_leaves = leafElements( tp_cmap );
    CHECK_NOTHROW( topology::detail::levelDomains( tp_cmap, tp_leaves ) );
    requireThrows( [&] { topology::detail::levelDomains( tp_cmap, { tp_leaves.begin(), tp_leaves.end() - 1 } ); },
                   "invalid_argument" );
    auto edge_leaves = tp_leaves;
    REQUIRE_FALSE( edge_leaves.front().empty() );
    edge_leaves.front().front() = Cell( edge_leaves.front().front().dart(), 1 );
    requireThrows( [&] { topology::detail::levelDomains( tp_cmap, edge_leaves ); }, "invalid_argument" );

    spec.patches = 2;
    const auto multi_fixture = fixtureLevels( spec );
    std::vector<std::shared_ptr<const MultiPatchSplineSpace>> multi_levels;
    for( const auto& patches : multi_fixture.patches )
        multi_levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
            buildH1MultiPatchSplineSpace( patches, multi_fixture.connections ) ) );
    const auto multi = buildHierarchicalSplineSpace( multi_levels, multiLeaves( multi_fixture, multi_levels ) );
    const auto& mp_cmap = multi.basisComplex().parametricAtlas().cmap();
    auto mp_leaves = leafElements( mp_cmap );
    CHECK_NOTHROW( topology::detail::levelDomains( mp_cmap, mp_leaves ) );
    auto extra = mp_leaves;
    extra.emplace_back();
    requireThrows( [&] { topology::detail::levelDomains( mp_cmap, extra ); }, "invalid_argument" );
    REQUIRE_FALSE( mp_leaves.back().empty() );
    mp_leaves.back().back() = Cell( mp_leaves.back().back().dart(), 1 );
    requireThrows( [&] { topology::detail::levelDomains( mp_cmap, mp_leaves ); }, "invalid_argument" );

    const auto line = bspline1d( { 0, 0, 0, 1, 2, 3, 3, 3 }, 2 );
    const SupportRanges1d ranges( *line );
    requireThrows( [&] { ranges.range( ranges.numFunctions() ); }, "out_of_range" );
    requireThrows( [] {
        const SupportRanges1d bad( *bspline1d( { 0, 0, 0, 1, 2, 3, 3, 3 }, 2, 2 ) );
        (void)bad;
    }, "logic_error" );

    const TensorSupport support( *levels.front() );
    const FunctionId past( support.numFunctions() );
    for( const FunctionId outside : { past, FunctionId( -1 ) } )
    {
        CAPTURE( outside.id() );
        requireThrows( [&] { support.ranges( outside ); }, "out_of_range" );
        requireThrows( [&] { support.iterateCellsWhile( outside, []( const auto& ) { return true; } ); },
                       "out_of_range" );
    }
    const auto& level0 = *levels.front();
    const TPSplineSpace wrapped( level0.basisComplexPtr(), std::make_shared<const WrappedSpace>( level0.sourcePtr() ),
                                 level0.linePtr() );
    REQUIRE( tensorProductComponentSplines( wrapped ).empty() );
    requireThrows( [&] {
        const TensorSupport bad( wrapped );
        (void)bad;
    }, "invalid_argument" );
}

TEST_CASE( "Self-glued patches are admitted with same-patch representatives", "[thb_queries]" )
{
    using Side = MultiPatchCombinatorialMap::ConstituentSide;
    struct Case
    {
        size_t dim;
        Permutation permutation;
        std::vector<std::array<size_t, 3>> levels; // local, global, globals with two local representatives
        size_t h1, hdiv, l2;
    };
    const std::vector<std::array<size_t, 3>> levels2{ { 16, 12, 4 }, { 36, 30, 6 }, { 100, 90, 10 } };
    const std::vector<std::array<size_t, 3>> levels3{ { 36, 27, 9 }, { 96, 80, 16 }, { 360, 324, 36 } };
    const std::vector<Case> cases{ { 2, Permutation::Flip1d, levels2, 54, 111, 55 },
                                   { 3, Permutation::ZeroToZero, levels3, 221, 556, 169 },
                                   { 3, Permutation::ZeroToOne, levels3, 224, 559, 169 },
                                   { 3, Permutation::ZeroToTwo, levels3, 221, 556, 169 },
                                   { 3, Permutation::ZeroToThree, levels3, 224, 559, 169 } };
    for( const Case& c : cases )
    {
        CAPTURE( c.dim, static_cast<int>( c.permutation ) );
        FixtureSpec spec;
        spec.dim = c.dim;
        if( c.dim == 3 )
        {
            spec.cells = { 2, 1, 1 };
            spec.ratios = { { 2, 2, 2 }, { 2, 2, 2 } };
        }
        auto fixture = fixtureLevels( spec );
        fixture.connections.emplace( Side{ 0, 0 }, std::make_pair( c.permutation, Side{ 0, 1 } ) );
        fixture.connections.emplace( Side{ 0, 1 }, std::make_pair( c.permutation, Side{ 0, 0 } ) );
        std::vector<std::shared_ptr<const MultiPatchSplineSpace>> levels;
        for( const auto& patches : fixture.patches )
            levels.push_back( std::make_shared<const MultiPatchSplineSpace>(
                buildH1MultiPatchSplineSpace( patches, fixture.connections ) ) );
        REQUIRE( levels.size() == c.levels.size() );
        for( size_t level = 0; level < levels.size(); level++ )
        {
            const auto& map = levels.at( level )->functionIdMap().at( 0 );
            std::map<size_t, size_t> uses;
            for( const auto& global : map ) uses[global.id()]++;
            size_t shared = 0;
            for( const auto& [global, n] : uses ) shared += n == 2;
            CAPTURE( level );
            const std::array<size_t, 3> measured{ map.size(), levels.at( level )->numFunctions(), shared };
            CHECK( measured == c.levels.at( level ) );
            CHECK( std::all_of( uses.begin(), uses.end(), []( const auto& u ) { return u.second <= 2; } ) );
        }
        const auto h1 = buildHierarchicalSplineSpace( levels, multiLeaves( fixture, levels ) );
        const auto hdiv = buildHDivHierarchicalMultiPatchSplineSpace( h1 );
        const auto hcurl = buildHCurlHierarchicalMultiPatchSplineSpace( h1 );
        const auto l2 = buildL2HierarchicalMultiPatchSplineSpace( hdiv );
        ActivityReport report;
        CHECK_NOTHROW( checkActivity( h1, "H1", report ) );
        CHECK_NOTHROW( checkActivity( hdiv, "Hdiv", report ) );
        CHECK_NOTHROW( checkActivity( hcurl, "Hcurl", report ) );
        CHECK_NOTHROW( checkActivity( l2, "L2", report ) );
        CHECK( report.levels == 12 );
        CHECK( h1.numFunctions() == c.h1 );
        CHECK( hdiv.numFunctions() == c.hdiv );
        CHECK( l2.numFunctions() == c.l2 );
        Counts counts;
        CHECK_NOTHROW( checkRegion( h1.basisComplex().parametricAtlas().cmap(), fixture, "self-glued", counts ) );
    }
}
