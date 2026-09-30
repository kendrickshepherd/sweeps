#include <SideCoordinateTransform.hpp>
#include <stdexcept>

namespace topology
{
    using TPPermutation = MultiPatchCombinatorialMap::TPPermutation;

    namespace
    {
        // F_s, restated from flipNecessaryDarts.
        bool sideReversesFirstAxis( const size_t dim, const size_t side_id )
        {
            if( dim == 2 ) return side_id == 1 or side_id == 2;
            return side_id == 0 or side_id == 3 or side_id == 4;
        }

        struct AxisImage
        {
            size_t axis;
            bool flip;
        };

        // P, restated from permuteAcrossBoundary as the image of a source axis.
        AxisImage permutationImage( const TPPermutation permutation, const size_t source_axis )
        {
            switch( permutation )
            {
                case TPPermutation::Flip1d: return { 0, true };
                case TPPermutation::ZeroToZero:
                    return source_axis == 0 ? AxisImage{ 0, true } : AxisImage{ 1, false };
                case TPPermutation::ZeroToOne:
                    return source_axis == 0 ? AxisImage{ 1, true } : AxisImage{ 0, true };
                case TPPermutation::ZeroToTwo:
                    return source_axis == 0 ? AxisImage{ 0, false } : AxisImage{ 1, true };
                case TPPermutation::ZeroToThree:
                    return source_axis == 0 ? AxisImage{ 1, false } : AxisImage{ 0, false };
            }
            throw std::invalid_argument( "Unknown tensor-product interface permutation." );
        }

        void validateTransform( const SideCoordinateTransform& transform )
        {
            const size_t rank = transform.source_axis_for_destination.size();
            if( rank != transform.source_axis_reversed.size() )
                throw std::invalid_argument( "Side coordinate transform axis and reversal ranks disagree." );
            if( rank != 1 and rank != 2 )
                throw std::invalid_argument( "Side coordinate transform must have rank one or two." );

            bool seen[2] = { false, false };
            for( size_t d = 0; d < rank; d++ )
            {
                const size_t s = transform.source_axis_for_destination.at( d );
                if( s >= rank or seen[s] )
                    throw std::invalid_argument( "Side coordinate transform is not a permutation of the "
                                                 "tangential axes." );
                seen[s] = true;
            }
        }

        void validateExtents( const SideCoordinateTransform& transform, const std::vector<size_t>& extents )
        {
            validateTransform( transform );
            if( extents.size() != transform.source_axis_for_destination.size() )
                throw std::invalid_argument( "Side extents do not match the transform rank." );
            for( const size_t n : extents )
                if( n == 0 ) throw std::invalid_argument( "Side extents must be positive." );
        }
    }

    SideCoordinateTransform sideCoordinateTransform( const size_t dim,
                                                    const size_t source_side_id,
                                                    const size_t destination_side_id,
                                                    const TPPermutation permutation )
    {
        if( dim != 2 and dim != 3 )
            throw std::invalid_argument( "Strong multipatch coupling is supported only in 2D and 3D." );
        if( source_side_id >= 2 * dim or destination_side_id >= 2 * dim )
            throw std::invalid_argument( "Patch interface side is outside the patch dimension." );
        if( dim == 2 and permutation != TPPermutation::Flip1d )
            throw std::invalid_argument( "A two-dimensional patch interface requires the Flip1d permutation." );
        if( dim == 3 and permutation == TPPermutation::Flip1d )
            throw std::invalid_argument( "A three-dimensional patch interface requires a face permutation." );

        const size_t rank = dim - 1;
        size_t source_for_destination[2] = { 0, 0 };
        bool reversed_for_destination[2] = { false, false };

        for( size_t source = 0; source < rank; source++ )
        {
            // F_a reverses source axis 0, then P permutes, then F_b reverses
            // destination axis 0.  Reversals compose by parity.
            bool reversed = source == 0 and sideReversesFirstAxis( dim, source_side_id );
            const AxisImage image = permutationImage( permutation, source );
            reversed = reversed != image.flip;
            if( image.axis == 0 and sideReversesFirstAxis( dim, destination_side_id ) ) reversed = not reversed;

            source_for_destination[image.axis] = source;
            reversed_for_destination[image.axis] = reversed;
        }

        SideCoordinateTransform out;
        for( size_t d = 0; d < rank; d++ )
        {
            out.source_axis_for_destination.push_back( source_for_destination[d] );
            out.source_axis_reversed.push_back( reversed_for_destination[d] );
        }
        validateTransform( out );
        return out;
    }

    std::vector<size_t> transformedExtents( const SideCoordinateTransform& transform,
                                            const std::vector<size_t>& source_extents )
    {
        validateExtents( transform, source_extents );
        std::vector<size_t> out;
        out.reserve( source_extents.size() );
        for( size_t d = 0; d < transform.source_axis_for_destination.size(); d++ )
            out.push_back( source_extents.at( transform.source_axis_for_destination.at( d ) ) );
        return out;
    }

    std::vector<size_t> transformSideIndex( const SideCoordinateTransform& transform,
                                           const std::vector<size_t>& source_index,
                                           const std::vector<size_t>& source_extents )
    {
        validateExtents( transform, source_extents );
        if( source_index.size() != source_extents.size() )
            throw std::invalid_argument( "Side index does not match the transform rank." );

        std::vector<size_t> out;
        out.reserve( source_index.size() );
        for( size_t d = 0; d < transform.source_axis_for_destination.size(); d++ )
        {
            const size_t s = transform.source_axis_for_destination.at( d );
            const size_t n = source_extents.at( s );
            const size_t i = source_index.at( s );
            if( i >= n ) throw std::invalid_argument( "Side index is outside the side extents." );
            out.push_back( transform.source_axis_reversed.at( d ) ? n - 1 - i : i );
        }
        return out;
    }

    std::vector<double> transformSidePoint( const SideCoordinateTransform& transform,
                                            const std::vector<double>& source_point )
    {
        validateTransform( transform );
        if( source_point.size() != transform.source_axis_for_destination.size() )
            throw std::invalid_argument( "Side point does not match the transform rank." );

        std::vector<double> out;
        out.reserve( source_point.size() );
        for( size_t d = 0; d < transform.source_axis_for_destination.size(); d++ )
        {
            const double x = source_point.at( transform.source_axis_for_destination.at( d ) );
            out.push_back( transform.source_axis_reversed.at( d ) ? 1.0 - x : x );
        }
        return out;
    }
}
