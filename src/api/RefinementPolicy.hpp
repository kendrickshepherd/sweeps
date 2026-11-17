#pragma once
#include <NavierStokesDiscretization.hpp>
#include <IndexOperations.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace api
{
    enum class RefinementPolicy
    {
        NFormSupport,
        ZeroFormSupport,
        EM,
        MHD
    };

    /// @brief Where a hierarchy first fails a policy.
    struct PolicyFailure
    {
        RefinementPolicy policy;
        size_t patch;
        size_t level;
        /// Level-local tensor cell indices, one per parametric direction.
        util::IndexVec cell;

        bool operator==( const PolicyFailure& ) const = default;
    };

    struct PolicyCheck
    {
        /// Empty iff the policy holds.
        std::optional<PolicyFailure> failure;

        bool satisfied() const { return not failure.has_value(); }
    };

    // The exceptions have default visibility so hidden-visibility modules (splines) catch them by type.

    /// @brief requireRefinementPolicy's refusal; failure() is the record checkRefinementPolicy returns.
    class __attribute__( ( visibility( "default" ) ) ) PolicyViolation : public std::runtime_error
    {
        public:
        explicit PolicyViolation( const PolicyFailure& failure );
        const PolicyFailure& failure() const { return mFailure; }

        private:
        PolicyFailure mFailure;
    };

    /// @brief A policy, or a valid space for a policy, that the checks do not implement.
    class __attribute__( ( visibility( "default" ) ) ) PolicyNotImplemented : public std::logic_error
    {
        public:
        using std::logic_error::logic_error;
    };

    /// @throws std::invalid_argument if policy is not a RefinementPolicy value.
    std::string policyName( const RefinementPolicy policy );

    /// @brief The diagnostic form.  NFormSupport is basis::checkNFormSupportPolicy on the L2 space; no policy falls
    /// back to another.
    /// @throws PolicyNotImplemented for ZeroFormSupport, EM and MHD, before any check, and for a space the
    ///         policy's check does not implement.
    /// @throws std::invalid_argument if policy is not a RefinementPolicy value.
    PolicyCheck checkRefinementPolicy( const NavierStokesHierarchicalDiscretization& disc, const RefinementPolicy policy );
    PolicyCheck checkRefinementPolicy( const NavierStokesHierarchicalMultiPatchDiscretization& disc,
                                       const RefinementPolicy policy );

    /// @brief The enforcing form of checkRefinementPolicy.
    /// @throws PolicyViolation carrying checkRefinementPolicy's failure; otherwise as checkRefinementPolicy.
    void requireRefinementPolicy( const NavierStokesHierarchicalDiscretization& disc, const RefinementPolicy policy );
    void requireRefinementPolicy( const NavierStokesHierarchicalMultiPatchDiscretization& disc,
                                  const RefinementPolicy policy );
}
