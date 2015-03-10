#!/bin/bash
class_index=0
for i in $( ls ); do
  ls | grep $i
  cd $i
  echo $i $class_index >> '../../class_list.txt'
  for j in $(ls | grep -v 'file_list.txt'); do
    echo $j $class_index >> '../../filenames.txt' 
  done
  let class_index=class_index+1
  cd ..
done
