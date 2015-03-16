set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5 
set title "Validation accuracy w.r.t. epochs"
set ylabel "Validation softmax accuracy"
set xlabel "Number of epochs"
set xrange [0:5]
#set yrange [0:0.5]
set term png
set output 'data_aug.png'
plot 'aug_accVsEpoch' using 1:2 with linespoint title "5-layer Baseline" , \
  'aug_accVsEpoch' using 1:3 with linespoint title "Data Augmentation" 
