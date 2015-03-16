set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5 
set title "Validation loss w.r.t. methods"
set ylabel "Validation accuracy"
set xlabel "Methods"
set xrange [0:6]
#set yrange [0:1]
set term png
set output 'lossVsMethod.png'
plot 'lossVsMethod' using 1:2, \
  'lossVsMethod' using 1:3, \
 'lossVsMethod' using 1:4


