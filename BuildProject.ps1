#Add external library directory to the linker
$ENV:LIB="external"

cargo +nightly run
pause