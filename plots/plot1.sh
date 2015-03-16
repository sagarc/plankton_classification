set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5 
set title "Validation accuracy w.r.t. epochs"
set ylabel "Validation accuracy"
set xlabel "Number of epochs"
set xrange [0:6]
set yrange [0:1]
set term png
set output 'accuracyVsEpoch.png'
plot 'accVsEpoch' using 1:2 with linespoint title "2-layer" , \
  'accVsEpoch' using 1:3 with linespoint title "3-layer" , \
 'accVsEpoch' using 1:4 with linespoint title "5-layer"


