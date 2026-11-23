[CmdletBinding()]
param(
    [string] $BuildDir = (Join-Path ([IO.Path]::GetTempPath()) 'PADGain_row_tail_build'),
    [ValidatePattern('^sm_[0-9]{2,3}[a-z]?$')]
    [string] $Arch = 'sm_89',
    [string] $SourceFile = (Join-Path $PSScriptRoot 'bench.cu')
)

$ErrorActionPreference = 'Stop'
$source = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SourceFile)
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "CUDA source not found: $source"
}
$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio with MSVC x64 build tools and the Windows SDK.'
}

# CUDA 12.8 supports MSVC v143. Visual Studio can have a newer default toolset.
$toolsets = foreach ($install in (& $vswhere -all -products '*' -property installationPath)) {
    $toolsDir = Join-Path $install 'VC\Tools\MSVC'
    if (Test-Path -LiteralPath $toolsDir) {
        foreach ($tool in (Get-ChildItem -LiteralPath $toolsDir -Directory)) {
            if ($tool.Name -match '^14\.[34][0-9]\.\d+$' -and
                (Test-Path -LiteralPath (Join-Path $tool.FullName 'bin\Hostx64\x64\cl.exe'))) {
                [pscustomobject]@{
                    Version = [version]$tool.Name
                    Setup = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
                }
            }
        }
    }
}
$selected = $toolsets | Sort-Object Version -Descending | Select-Object -First 1
if (-not $selected) {
    throw 'Install the MSVC v143 x64 toolset for CUDA 12.8.'
}

$resolvedBuildDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BuildDir)
$path = [IO.Directory]::CreateDirectory($resolvedBuildDir).FullName
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = $env:ComSpec
$start.UseShellExecute = $false
$start.WorkingDirectory = $path
$start.EnvironmentVariables['PADGAIN_SETUP'] = $selected.Setup
$start.EnvironmentVariables['PADGAIN_VC_VERSION'] = $selected.Version.ToString()
$start.EnvironmentVariables['PADGAIN_NVCC'] = $nvcc
$start.EnvironmentVariables['PADGAIN_ARCH'] = $Arch
$start.EnvironmentVariables['PADGAIN_SOURCE'] = $source
# Relative intermediate names also avoid nvcc rewriting '#' in temporary paths.
$start.Arguments = '/d /v:off /s /c "call "%PADGAIN_SETUP%" -vcvars_ver=%PADGAIN_VC_VERSION% && "%PADGAIN_NVCC%" --use-local-env --keep --keep-dir . -std=c++17 -O3 -lineinfo -arch=%PADGAIN_ARCH% "%PADGAIN_SOURCE%" -o bench.exe"'
$compiler = [Diagnostics.Process]::Start($start)
$compiler.WaitForExit()
if ($compiler.ExitCode -ne 0) {
    throw "CUDA build failed (exit code $($compiler.ExitCode))."
}
Write-Output (Join-Path $path 'bench.exe')
