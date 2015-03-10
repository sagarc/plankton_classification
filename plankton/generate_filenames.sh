#!/bin/bash
  for i in $( ls ); do
    ls | grep $i
    #cd $i
    #rm file_list.txt
    #cd ..
    cd $i
    ls | grep -v 'file_list.txt' > file_list.txt
    cd ..
  done
