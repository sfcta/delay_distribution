:: Directory where input files live
set RUNDIR=.

:: Directory where "generate_delay_distribution.py" sits.  Usually one up from this batch file.
set SCRIPTDIR=..

:: List of scenario names that input files are labelled with.
set ALLSCENARIOS=Scenario1 Scenario2 Scenario3

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set thisdate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set thistime=%%a%%b)
ping 1.1.1.1 -n 1 -w 100 > nul

for %%H in (%ALLSCENARIOS%) do (
    set THISSCENARIO=%%H
    start CallDelayDistScript.bat
    ping 1.1.1.1 -n 1 -w 500 > nul
)
