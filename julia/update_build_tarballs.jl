#!/usr/bin/env julia
"""
Update build_tarballs.jl with a new git tag.

Usage:
    julia update_build_tarballs.jl <tag>

Example:
    julia update_build_tarballs.jl v0.8.0
"""

using Pkg

function get_commit_hash(tag::String)
    """Get the full commit hash for a given tag."""
    result = read(`git rev-parse $tag`, String)
    return strip(result)
end

function parse_version(tag::String)
    """Parse version from tag (e.g., 'v0.8.0' -> '0.8.0')."""
    m = match(r"^v?(.+)$", tag)
    if m === nothing
        error("Invalid tag format: $tag")
    end
    return m.captures[1]
end

function update_build_tarballs(tag::String, build_file::String="build_tarballs.jl")
    """Update build_tarballs.jl with new version and commit hash."""
    
    # Get commit hash
    commit_hash = get_commit_hash(tag)
    version_str = parse_version(tag)
    
    println("Tag: $tag")
    println("Version: $version_str")
    println("Commit: $commit_hash")
    
    # Read the file
    content = read(build_file, String)
    
    # Update version line
    content = replace(content, r"version = v\"[^\"]+\"" => "version = v\"$version_str\"")
    
    # Update git commit hash (the second argument of GitSource)
    content = replace(content, 
        r"(GitSource\(\s*\"[^\"]+\",\s*)\"[a-f0-9]+\"" => 
        SubstitutionString("\\1\"$commit_hash\""))
    
    # Update comment with tag reference
    content = replace(content,
        r"# sparse-ir-rs v[^\n]+" =>
        "# sparse-ir-rs $tag")
    
    # Write back
    write(build_file, content)
    
    println("\n✅ Updated $build_file")
    println("   - version = v\"$version_str\"")
    println("   - commit = \"$commit_hash\"")
end

function main()
    if length(ARGS) != 1
        println(stderr, "Error: Missing tag argument")
        println(stderr, "Usage: julia update_build_tarballs.jl <tag>")
        println(stderr, "Example: julia update_build_tarballs.jl v0.8.0")
        exit(1)
    end
    
    tag = ARGS[1]
    
    # Change to the git repository root
    repo_root = chomp(read(`git rev-parse --show-toplevel`, String))
    cd(repo_root)
    
    # Check if tag exists
    try
        run(`git rev-parse $tag`)
    catch
        println(stderr, "Error: Tag '$tag' does not exist")
        exit(1)
    end
    
    # Update build_tarballs.jl
    build_file = joinpath("julia", "build_tarballs.jl")
    if !isfile(build_file)
        println(stderr, "Error: $build_file not found")
        exit(1)
    end
    
    update_build_tarballs(tag, build_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

