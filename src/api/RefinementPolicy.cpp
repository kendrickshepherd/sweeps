#include <RefinementPolicy.hpp>
#include <NFormSupportPolicy.hpp>

namespace api
{
    namespace
    {
        std::string describe( const PolicyFailure& f )
        {
            std::string cell = "(";
            for( size_t i = 0; i < f.cell.size(); i++ ) cell += ( i == 0 ? "" : ", " ) + std::to_string( f.cell.at( i ) );
            return "Refinement policy " + policyName( f.policy ) + " is violated on patch " + std::to_string( f.patch ) +
                   ", level " + std::to_string( f.level ) + ", cell " + cell + ").";
        }

        template <typename L2Space>
        PolicyCheck check( const L2Space& l2, const RefinementPolicy policy )
        {
            switch( policy )
            {
                case RefinementPolicy::NFormSupport:
                {
                    std::optional<basis::NFormSupportFailure> f;
                    try
                    {
                        f = basis::checkNFormSupportPolicy( l2 );
                    }
                    catch( const basis::NFormSupportNotImplemented& e )
                    {
                        throw PolicyNotImplemented( e.what() );
                    }
                    if( not f ) return {};
                    return { PolicyFailure{ policy, f->patch, f->level, f->cell } };
                }
                case RefinementPolicy::ZeroFormSupport:
                case RefinementPolicy::EM:
                case RefinementPolicy::MHD:
                    throw PolicyNotImplemented( "Refinement policy " + policyName( policy ) + " is not implemented." );
            }
            throw std::invalid_argument( "Unknown refinement policy." );
        }

        void require( const PolicyCheck& c )
        {
            if( c.failure ) throw PolicyViolation( *c.failure );
        }
    }

    PolicyViolation::PolicyViolation( const PolicyFailure& failure )
        : std::runtime_error( describe( failure ) ), mFailure( failure )
    {}

    std::string policyName( const RefinementPolicy policy )
    {
        switch( policy )
        {
            case RefinementPolicy::NFormSupport: return "NFormSupport";
            case RefinementPolicy::ZeroFormSupport: return "ZeroFormSupport";
            case RefinementPolicy::EM: return "EM";
            case RefinementPolicy::MHD: return "MHD";
        }
        throw std::invalid_argument( "Unknown refinement policy." );
    }

    PolicyCheck checkRefinementPolicy( const NavierStokesHierarchicalDiscretization& disc, const RefinementPolicy policy )
    {
        return check( disc.L2_ss, policy );
    }

    PolicyCheck checkRefinementPolicy( const NavierStokesHierarchicalMultiPatchDiscretization& disc,
                                       const RefinementPolicy policy )
    {
        return check( disc.L2_ss, policy );
    }

    void requireRefinementPolicy( const NavierStokesHierarchicalDiscretization& disc, const RefinementPolicy policy )
    {
        require( checkRefinementPolicy( disc, policy ) );
    }

    void requireRefinementPolicy( const NavierStokesHierarchicalMultiPatchDiscretization& disc,
                                  const RefinementPolicy policy )
    {
        require( checkRefinementPolicy( disc, policy ) );
    }
}
