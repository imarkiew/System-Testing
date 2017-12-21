#!/usr/bin/env bash
directory_path=$PWD
Py_interpreter='/usr/bin/python3.6'
R_interpreter='Rscript'
name_of_data_file='iris.data'
number_of_iterations=5
is_category_numerical=false
num_of_labels=3
is_header_present=false
name_or_number_of_target_column=5
separator=','
percent_of_test_examples=0.3
is_oversampling_enabled=false
for((i=1;i<=number_of_iterations;i++))
do
   $Py_interpreter -m PyFiles.DivideSet $name_of_data_file $directory_path $is_header_present $name_or_number_of_target_column \
                                        $separator $percent_of_test_examples $is_oversampling_enabled
   $Py_interpreter -m PyFiles.Test $directory_path
   $R_interpreter ./RFiles/Test.R $directory_path $is_category_numerical $num_of_labels
   $Py_interpreter -m PyFiles.ProcessRResults $directory_path
done
$Py_interpreter -m PyFiles.ProcessPyResults $number_of_iterations $directory_path
$Py_interpreter -m PyFiles.FindFinalPyStats $directory_path
$Py_interpreter -m PyFiles.FindFinalRStats $directory_path
./ClearPyResults.sh
./ClearRResults.sh