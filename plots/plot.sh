set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5 
set title "Validation loss w.r.t. epochs"
set ylabel "Validation softmax loss"
set xlabel "Number of epochs"
set xrange [0:7]
#set yrange [0:0.5]
set terminal png
set output 'lossVsEpoch.png'
plot 'lossVsEpoch' using 1:2 with linespoint title "2-layer" , \
  'lossVsEpoch' using 1:3 with linespoint title "3-layer" , \
 'lossVsEpoch' using 1:4 with linespoint title "5-layer"


