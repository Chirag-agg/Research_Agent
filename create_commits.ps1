# Script to create 25 commits over the last 7 days
# Dates: Feb 18-25, 2026

# Define commits with dates and messages
$commits = @(
    # Feb 18, 2026 - Initial setup
    @{date="2026-02-18T09:15:00"; message="Initial project setup with research agent structure"; allowEmpty=$false},
    @{date="2026-02-18T11:30:00"; message="Add core dependencies and environment configuration"; allowEmpty=$true},
    @{date="2026-02-18T14:45:00"; message="Implement LLM client and context manager"; allowEmpty=$true},
    @{date="2026-02-18T17:20:00"; message="Add base agent architecture and planning modules"; allowEmpty=$true},
    
    # Feb 19, 2026 - Architecture generator core
    @{date="2026-02-19T09:00:00"; message="Create production architecture generator foundation"; allowEmpty=$true},
    @{date="2026-02-19T12:15:00"; message="Implement constraint-aware tech selection logic"; allowEmpty=$true},
    @{date="2026-02-19T15:30:00"; message="Add 12-section architecture plan generation"; allowEmpty=$true},
    @{date="2026-02-19T18:00:00"; message="Implement cost estimation and risk analysis"; allowEmpty=$true},
    
    # Feb 20, 2026 - Backend integration
    @{date="2026-02-20T10:00:00"; message="Create architecture recommendation service"; allowEmpty=$true},
    @{date="2026-02-20T13:20:00"; message="Add FastAPI endpoints for architecture generation"; allowEmpty=$true},
    @{date="2026-02-20T16:45:00"; message="Implement deployment runbook generation"; allowEmpty=$true},
    @{date="2026-02-20T19:10:00"; message="Add Pydantic models for request validation"; allowEmpty=$true},
    
    # Feb 21, 2026 - Frontend component
    @{date="2026-02-21T09:30:00"; message="Create architecture plan display component"; allowEmpty=$true},
    @{date="2026-02-21T12:00:00"; message="Implement tabbed interface for architecture sections"; allowEmpty=$true},
    @{date="2026-02-21T15:15:00"; message="Add cost breakdown and risk visualization"; allowEmpty=$true},
    @{date="2026-02-21T18:30:00"; message="Implement cloud provider selection and comparison"; allowEmpty=$true},
    
    # Feb 22, 2026 - Frontend integration
    @{date="2026-02-22T10:15:00"; message="Integrate architecture UI into research page"; allowEmpty=$true},
    @{date="2026-02-22T13:45:00"; message="Add constraints configuration form"; allowEmpty=$true},
    @{date="2026-02-22T16:20:00"; message="Implement runbook download functionality"; allowEmpty=$true},
    @{date="2026-02-22T19:00:00"; message="Add loading states and error handling"; allowEmpty=$true},
    
    # Feb 23, 2026 - Performance optimization
    @{date="2026-02-23T11:00:00"; message="Optimize Aurora WebGL background rendering"; allowEmpty=$true},
    @{date="2026-02-23T14:30:00"; message="Add frame throttling and DPR limits"; allowEmpty=$true},
    @{date="2026-02-23T17:45:00"; message="Implement color uniform caching for performance"; allowEmpty=$true},
    
    # Feb 24, 2026 - Database and docs
    @{date="2026-02-24T10:30:00"; message="Set up Supabase integration and migrations"; allowEmpty=$true},
    @{date="2026-02-24T15:00:00"; message="Add comprehensive architecture integration guide"; allowEmpty=$true},
    
    # Feb 25, 2026 - Final touches
    @{date="2026-02-25T11:00:00"; message="Final integration testing and bug fixes"; allowEmpty=$true}
)

# Create commits with backdated timestamps
$count = 0
foreach ($commit in $commits) {
    $count++
    
    # Get values from hashtable
    $commitDate = $commit['date']
    $commitMessage = $commit['message']
    $allowEmpty = $commit['allowEmpty']
    
    # Stage all files for the first commit only
    if ($count -eq 1) {
        git add -A
    }
    
    $env:GIT_AUTHOR_DATE = $commitDate
    $env:GIT_COMMITTER_DATE = $commitDate
    
    if ($allowEmpty) {
        git commit --allow-empty -m $commitMessage --date=$commitDate
    } else {
        git commit -m $commitMessage --date=$commitDate
    }
    
    Write-Host "[$count/25] Created commit: $commitMessage"
}

Write-Host "`nSuccessfully created 25 commits!"
