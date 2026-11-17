#include <catch2/catch_test_macros.hpp>
#include <BSplineSpace1d.hpp>
#include <BasisComplex1d.hpp>
#include <ParametricAtlas1d.hpp>
#include <CombinatorialMap1d.hpp>
#include <KnotVector.hpp>
#include <CombinatorialMapMethods.hpp>
#include <Logging.hpp>
#include <CommonUtils.hpp>
#include <TPSplineSpace.hpp>
#include <Eigen/Dense>
#include <set>

using namespace basis;
using namespace param;
using namespace topology;

TEST_CASE( "Extraction of cubic spline with various knot multiplicities" )
{
    const KnotVector kv( {0,0,0,0,1,1,2,3,3,3,3}, 1e-10 );
    const size_t degree = 3;

    const auto cmap = std::make_shared<const CombinatorialMap1d>( 3 );
    const auto param = std::make_shared<const ParametricAtlas1d>( cmap, parametricLengths( kv ) );
    const auto bc = std::make_shared<const BasisComplex1d>( param, degree );
    const BSplineSpace1d ss( bc, kv );

    iterateCellsWhile( *cmap, 1, [&]( const topology::Edge& e ) {
        const Eigen::MatrixXd op = ss.extractionOperator( e );
        CHECK( util::equals( op.colwise().sum(), Eigen::VectorXd::Ones( op.cols() ), 1e-12 ) );
        CHECK( ss.connectivity( e ).size() == degree + 1 );
        CHECK( op.rows() == degree + 1 );
        CHECK( op.rows() == op.cols() );

        Eigen::FullPivLU<Eigen::MatrixXd> lu( op );
        CHECK( lu.rank() == op.rows() );
        return true;
    } );
}

TEST_CASE( "Spline function counts and numbering survive caching", "[spline_count]" )
{
    for( size_t degree = 0; degree <= 4; degree++ )
    {
        for( size_t multiplicity = 1; multiplicity <= degree + 1; multiplicity++ )
        {
            CAPTURE( degree, multiplicity );
            // Nonuniform spans, including smooth, C0, broken and mixed multiplicities.
            const std::vector<double> breaks{ -2.0, -1.75, 0.5, 3.0 };
            std::vector<double> knots( degree + 1, breaks.front() );
            knots.insert( knots.end(), multiplicity, breaks.at( 1 ) );
            knots.insert( knots.end(), degree + 2 - multiplicity, breaks.at( 2 ) );
            knots.insert( knots.end(), degree + 1, breaks.back() );
            KnotVector kv( knots, 1e-12 );
            const auto cmap = std::make_shared<const CombinatorialMap1d>( breaks.size() - 1 );
            const auto atlas = std::make_shared<const ParametricAtlas1d>( cmap, parametricLengths( kv ) );
            const auto bc = std::make_shared<const BasisComplex1d>( atlas, degree );
            const BSplineSpace1d ss( bc, kv );
            const size_t expected_count = knots.size() - degree - 1;
            CHECK( ss.numFunctions() == expected_count );

            std::set<size_t> seen;
            iterateCellsWhile( *cmap, 1, [&]( const Cell& cell ) {
                const size_t elem = cell.dart().id();
                const double midpoint = 0.5 * ( breaks.at( elem ) + breaks.at( elem + 1 ) );
                std::vector<FunctionId> expected;
                for( size_t fid = 0; fid < expected_count; fid++ )
                    if( knots.at( fid ) < midpoint and midpoint < knots.at( fid + degree + 1 ) )
                        expected.emplace_back( fid );
                CHECK( ss.connectivity( cell ) == expected );
                for( const FunctionId& fid : ss.connectivity( cell ) ) seen.insert( fid.id() );
                return true;
            } );
            CHECK( seen.size() == expected_count );
            CHECK( *seen.begin() == 0 );
            CHECK( *seen.rbegin() == expected_count - 1 );

            // A spline owns its knot vector; later caller mutations must not alter its cached count.
            kv.duplicate( 0 );
            kv.insert( -1.0, 1e-12 );
            CHECK( kv.size() == knots.size() + 2 );
            CHECK( ss.knotVector().size() == knots.size() );
            CHECK( ss.numFunctions() == expected_count );
            const BSplineSpace1d copied( ss );
            CHECK( copied.numFunctions() == expected_count );
        }
    }
}

TEST_CASE( "TP Spline space" )
{
    const KnotVector kv1( {0,0,0,0,1,2,2,2,2}, 1e-10 );
    const KnotVector kv2( {0,0,0,1,2,2,2}, 1e-10 );
    const size_t degree1 = 3;
    const size_t degree2 = 2;

    const auto cmap = std::make_shared<const CombinatorialMap1d>( 2 );
    const auto param = std::make_shared<const ParametricAtlas1d>( cmap, parametricLengths( kv1 ) );
    const auto bc1 = std::make_shared<const BasisComplex1d>( param, degree1 );
    const auto bc2 = std::make_shared<const BasisComplex1d>( param, degree2 );
    const auto ss1 = std::make_shared<const BSplineSpace1d>( bc1, kv1 );
    const auto ss2 = std::make_shared<const BSplineSpace1d>( bc2, kv2 );

    const auto cmap_2d = std::make_shared<const TPCombinatorialMap>( cmap, cmap );
    const auto param_2d = std::make_shared<const TPParametricAtlas>( cmap_2d, param, param );
    const auto bc_2d = std::make_shared<const TPBasisComplex>( param_2d, bc1, bc2 );
    const TPSplineSpace ss_2d( bc_2d, ss1, ss2 );

    iterateCellsWhile( *cmap_2d, 2, [&]( const topology::Face& f ) {
        const Eigen::MatrixXd op = ss_2d.extractionOperator( f );
        CHECK( util::equals( op.colwise().sum(), Eigen::VectorXd::Ones( op.cols() ), 1e-12 ) );
        CHECK( ss_2d.connectivity( f ).size() == ( degree1 + 1 ) * ( degree2 + 1 ) );
        CHECK( op.rows() == ( degree1 + 1 ) * ( degree2 + 1 ) );
        CHECK( op.rows() == op.cols() );

        Eigen::FullPivLU<Eigen::MatrixXd> lu( op );
        CHECK( lu.rank() == op.rows() );
        return true;
    } );
}
