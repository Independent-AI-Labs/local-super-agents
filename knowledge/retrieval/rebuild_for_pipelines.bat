@echo off
call conda activate pipelines

REM Build the Python extension in place
python setup.py build_ext --inplace

REM Move the compiled .pyd file to the desired folder
move line_indexing.cp* hype\indexing\

REM Display success message
echo Build and move operation completed successfully!
