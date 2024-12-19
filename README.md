# 初步運行結果:
\begin{table}[]
\begin{tabular}{cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}
CIFAR100-LT                &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
                           &          &                 &                &                  &                 &                    &                   & \multicolumn{54}{c}{diverse test class distributions}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \\
                           & \multicolumn{7}{c}{uniform distribution}                                                                                  & \multicolumn{4}{c}{forward50}              &  & \multicolumn{4}{c}{forward25}              &  & \multicolumn{4}{c}{forward10}              &  & \multicolumn{4}{c}{forward5}               &  & \multicolumn{4}{c}{forward2}               &  & \multicolumn{4}{c}{uniform}                &  & \multicolumn{4}{c}{backward2}              &  & \multicolumn{4}{c}{backward5}              &  & \multicolumn{4}{c}{backward10}             &  & \multicolumn{4}{c}{backward25}             &  & \multicolumn{4}{c}{backward50}             \\
Standard-training   regime & accruacy & few\_class\_num & few\_shot\_acc & many\_class\_num & many\_shot\_acc & medium\_class\_num & medium\_shot\_acc & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   &  & Many-shot & Medium-shot & Few-shot & All   \\
SEED=1                     & 0.5169   & 30              & 0.31399998     & 35               & 0.67200005      & 35                 & 0.5357144         & 67.01     & 51.77       & 38       & 64.52 &  & 67.16     & 52.11       & 36.36    & 62.96 &  & 67.1      & 53.45       & 33.04    & 60.58 &  & 67.21     & 52.88       & 30.53    & 57.87 &  & 67.32     & 53.07       & 32.16    & 54.69 &  & 67.2      & 53.57       & 31.4     & 51.69 &  & 66.41     & 53.3        & 31.35    & 48.33 &  & 67.49     & 53.13       & 31.27    & 44.53 &  & 66.9      & 52.57       & 31.52    & 41.62 &  & 67.94     & 52.74       & 31.17    & 38.33 &  & 72.67     & 53.72       & 31.11    & 36.52 \\
SEED=2                     & 0.518    & 30              & 0.30466667     & 35               & 0.6788571       & 35                 & 0.53999996        & 67.85     & 56.26       & 32.61    & 65.65 &  & 67.38     & 55.85       & 28.99    & 63.6  &  & 67.35     & 54.69       & 30.22    & 60.89 &  & 67.79     & 53.54       & 29.65    & 58.33 &  & 67.84     & 52.82       & 31.09    & 54.62 &  & 67.89     & 54          & 30.47    & 51.8  &  & 67.15     & 52.97       & 30.65    & 48.13 &  & 69.08     & 53.36       & 30.63    & 44.55 &  & 69.57     & 53.71       & 30.6     & 41.77 &  & 65.65     & 54.19       & 30.49    & 38.09 &  & 71.75     & 54.14       & 30.74    & 36.28 \\
SEED=3                     & 0.5206   & 30              & 0.303          & 35               & 0.6745715       & 35                 & 0.55314285        & 67.65     & 54.44       & 29.33    & 65.04 &  & 67.02     & 54.55       & 27.62    & 62.99 &  & 66.67     & 54.77       & 28.8     & 60.32 &  & 67.04     & 53.79       & 28.78    & 57.87 &  & 67.33     & 54.5        & 30.27    & 54.77 &  & 67.46     & 55.31       & 30.3     & 52.06 &  & 66.71     & 55.33       & 30.59    & 48.81 &  & 66.56     & 54.27       & 30.57    & 44.32 &  & 66.84     & 54.25       & 30.87    & 41.62 &  & 65.22     & 54.84       & 29.94    & 37.75 &  & 68.34     & 54.85       & 29.83    & 35.68 \\
SEED=4                     & 0.5225   & 30              & 0.3086667      & 35               & 0.6831428       & 35                 & 0.5451428         & 69.09     & 55.21       & 31.94    & 66.09 &  & 68.52     & 54.07       & 31.64    & 63.94 &  & 68.15     & 54.5        & 30.76    & 61.15 &  & 68.25     & 53.78       & 29.73    & 58.45 &  & 68.45     & 54.16       & 30.68    & 55.16 &  & 68.31     & 54.51       & 30.87    & 52.26 &  & 68.08     & 53.97       & 30.93    & 48.91 &  & 68.92     & 54.18       & 31.17    & 45.26 &  & 68.19     & 53.44       & 30.8     & 41.92 &  & 66.4      & 53.89       & 30.3     & 38.33 &  & 68.96     & 54.36       & 29.99    & 36.2  \\
SEED=5                     & 0.5208   & 30              & 0.30966666     & 35               & 0.6714286       & 35                 & 0.5511429         & 66.68     & 55.45       & 29.56    & 64    &  & 65.85     & 54.91       & 32.29    & 62.55 &  & 65.91     & 55.14       & 32.1     & 60.37 &  & 66.58     & 54.78       & 30.55    & 58.17 &  & 66.88     & 54.46       & 31.78    & 54.88 &  & 67.14     & 55.11       & 50.97    & 52.08 &  & 65.67     & 54.83       & 30.99    & 48.49 &  & 65.82     & 55.15       & 30.93    & 44.67 &  & 66.56     & 54.08       & 31.13    & 41.74 &  & 63.56     & 53.82       & 30.85    & 37.96 &  & 65.22     & 54.79       & 30.89    & 36.24 \\
mean                       & 0.51976  & 30              & 0.308000002    & 35               & 0.67600001      & 35                 & 0.545028582       & 67.656    & 54.626      & 32.288   & 65.06 &  & 67.186    & 54.298      & 31.38    & 63.21 &  & 67.036    & 54.51       & 30.984   & 60.66 &  & 67.374    & 53.754      & 29.848   & 58.14 &  & 67.564    & 53.802      & 31.196   & 54.82 &  & 67.6      & 54.5        & 34.802   & 51.98 &  & 66.804    & 54.08       & 30.902   & 48.53 &  & 67.574    & 54.018      & 30.914   & 44.67 &  & 67.612    & 53.61       & 30.984   & 41.73 &  & 65.754    & 53.896      & 30.55    & 38.09 &  & 69.388    & 54.372      & 30.512   & 36.18 \\
standard   deviation       & 0.002268 & 0               & 0.004339732    & 0                & 0.00495279      & 0                  & 0.007336217       & 0.9305805 & 1.723464534 & 3.501553 & 0.84  &  & 0.9526699 & 1.38618902  & 3.374381 & 0.555 &  & 0.8284805 & 0.636513943 & 1.648963 & 0.353 &  & 0.6537048 & 0.68270052  & 0.73343  & 0.264 &  & 0.6006913 & 0.79819797  & 0.775455 & 0.211 &  & 0.4948232 & 0.731983606 & 9.048103 & 0.23  &  & 0.8941365 & 0.997196069 & 0.304171 & 0.326 &  & 1.4310416 & 0.805090057 & 0.312858 & 0.355 &  & 1.2626837 & 0.661626783 & 0.354584 & 0.124 &  & 1.6051106 & 0.761531352 & 0.47765  & 0.249 &  & 2.9582207 & 0.469648805 & 0.567908 & 0.308 \\
                           &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
long-training   regime     &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
SEED=1                     & 0.5546   & 30              & 0.3433332      & 35               & 0.7102857       & 35                 & 0.5774285         & 71.01     & 60.2        & 33       & 68.26 &  & 70.43     & 59.89       & 33.69    & 66.54 &  & 70.82     & 58.44       & 35.2     & 64.34 &  & 70.82     & 57.09       & 35.78    & 61,84 &  & 70.85     & 56.86       & 35.52    & 58.25 &  & 71.03     & 57.74       & 34.63    & 55.46 &  & 70.49     & 57.16       & 35.23    & 52.29 &  & 70.96     & 57.18       & 35.75    & 48.72 &  & 71.72     & 57.65       & 35.91    & 46.31 &  & 69.98     & 57.74       & 35.89    & 43.13 &  & 76.91     & 56.53       & 36.43    & 41.35 \\
SEED=2                     & 0.5575   & 30              & 0.35166666     & 35               & 0.71028566      & 35                 & 0.58114284        & 71.53     & 57.87       & 33.33    & 68.7  &  & 71.15     & 57.57       & 34.27    & 66.91 &  & 71        & 58.08       & 34.82    & 64.55 &  & 71.25     & 57.29       & 35.18    & 62.12 &  & 71.16     & 57.33       & 35.76    & 58.6  &  & 71.03     & 58.11       & 35.17    & 55.75 &  & 71.18     & 57.7        & 35.32    & 52.68 &  & 71.85     & 58.26       & 35.46    & 49.05 &  & 71.09     & 56.78       & 35.23    & 45.72 &  & 66.73     & 56.85       & 35.04    & 42.29 &  & 69.54     & 56.08       & 35.03    & 40.31 \\
SEED=3                     & 0.5529   & 30              & 0.353          & 35               & 0.7028571       & 35                 & 0.57428575        & 71.22     & 57.78       & 35.28    & 68.1  &  & 70.41     & 57.78       & 36.23    & 66.24 &  & 70.22     & 57.94       & 36.24    & 63.98 &  & 70.18     & 56.58       & 36.24    & 61.44 &  & 70.37     & 56.67       & 35.75    & 58.01 &  & 70.29     & 57.43       & 35.3     & 55.29 &  & 70.44     & 57.13       & 35.49    & 52.37 &  & 71.93     & 57.08       & 35.7     & 48.91 &  & 72.59     & 57.19       & 35.86    & 46.41 &  & 70.68     & 56.61       & 35.47    & 42.73 &  & 74.33     & 56.27       & 35.76    & 40.99 \\
SEED=4                     & 0.5598   & 30              & 0.36666667     & 35               & 0.7048572       & 35                 & 0.5802857         & 70.89     & 57.17       & 34.56    & 68.14 &  & 70.26     & 57.58       & 34.08    & 66.37 &  & 70.02     & 57.25       & 37.09    & 64.09 &  & 70.17     & 56.61       & 37.66    & 61.84 &  & 70.69     & 57.56       & 37.43    & 58.9  &  & 70.49     & 58.03       & 36.67    & 55.98 &  & 70.42     & 57.76       & 36.82    & 53    &  & 71.41     & 57.55       & 37.08    & 49.39 &  & 73.25     & 57.64       & 37.09    & 46.85 &  & 69.77     & 56.88       & 36.82    & 42.96 &  & 73.75     & 56.55       & 37.11    & 40.99 \\
SEED=5                     & 0.5553   & 30              & 0.36066666     & 35               & 0.7062857       & 35                 & 0.57114285        & 70.92     & 55.92       & 36.5     & 68.02 &  & 70.23     & 56.34       & 35.5     & 66.04 &  & 69.85     & 56.32       & 34.24    & 63.39 &  & 70.38     & 56.02       & 35.7     & 61.42 &  & 70.64     & 56.53       & 36.95    & 58.39 &  & 70.63     & 57.11       & 36.07    & 55.53 &  & 70.09     & 56.96       & 36.43    & 52.53 &  & 70.91     & 56.37       & 36.88    & 48.93 &  & 72.11     & 56.43       & 36.68    & 46.41 &  & 69.99     & 55.58       & 36.46    & 42.79 &  & 76.06     & 55.46       & 36.7     & 41.11 \\
mean                       & 0.55602  & 30              & 0.355066638    & 35               & 0.706914272     & 35                 & 0.576857128       & 71.114    & 57.788      & 34.534   & 68.24 &  & 70.496    & 57.832      & 34.754   & 66.42 &  & 70.382    & 57.606      & 35.518   & 64.07 &  & 70.56     & 56.718      & 36.112   & 61.71 &  & 70.742    & 56.99       & 36.282   & 58.43 &  & 70.694    & 57.684      & 35.568   & 55.6  &  & 70.524    & 57.342      & 35.858   & 52.57 &  & 71.412    & 57.288      & 36.174   & 49    &  & 72.152    & 57.138      & 36.154   & 46.34 &  & 69.43     & 56.732      & 35.936   & 42.78 &  & 74.118    & 56.178      & 36.206   & 40.95 \\
standard   deviation       & 0.002681 & 0               & 0.008936238    & 0                & 0.003309837     & 0                  & 0.004174741       & 0.2659511 & 1.556910402 & 1.433974 & 0.269 &  & 0.3761383 & 1.284161205 & 1.067207 & 0.329 &  & 0.5035077 & 0.838498658 & 1.141718 & 0.44  &  & 0.4670653 & 0.495852801 & 0.943568 & 0.338 &  & 0.2906372 & 0.4391469   & 0.85151  & 0.339 &  & 0.3296665 & 0.417827716 & 0.802571 & 0.268 &  & 0.3992869 & 0.362932501 & 0.719701 & 0.281 &  & 0.4786648 & 0.691209086 & 0.747248 & 0.248 &  & 0.8236626 & 0.535322333 & 0.733573 & 0.405 &  & 1.547918  & 0.773608428 & 0.720784 & 0.315 &  & 2.8594178 & 0.446060534 & 0.820567 & 0.387 \\
                           &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
more experts   (7)         &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
SEED=1                     & 0.5745   & 30              & 0.37700006     & 35               & 0.7254286       & 35                 & 0.5928572         & 72.48     & 60.97       & 34.11    & 69.75 &  & 71.76     & 59.67       & 33.31    & 67.63 &  & 71.99     & 59.14       & 35.4     & 65.48 &  & 72.3      & 58.91       & 38.23    & 63.65 &  & 72.62     & 59          & 38.65    & 60.45 &  & 72.54     & 59.29       & 37.7     & 57.45 &  & 72.38     & 59.29       & 38.16    & 54.62 &  & 72.85     & 59.26       & 38.62    & 50.99 &  & 74.8      & 59.46       & 38.43    & 48.5  &  & 71.43     & 58.76       & 38.91    & 45.03 &  & 76.27     & 58.35       & 38.98    & 43    \\
SEED=2                     & 0.5755   & 30              & 0.3773333      & 35               & 0.7288572       & 35                 & 0.592             & 73.43     & 61.83       & 36.11    & 70.96 &  & 73.13     & 61          & 35.94    & 69.11 &  & 72.81     & 59.99       & 37.95    & 66.56 &  & 73.07     & 58.96       & 38.5     & 64.17 &  & 73.04     & 58.77       & 38.97    & 60.66 &  & 72.89     & 59.2        & 37.73    & 57.55 &  & 73.05     & 58.99       & 38.26    & 54.76 &  & 73.99     & 59.65       & 39.15    & 51.66 &  & 74.89     & 59.95       & 39.01    & 49.12 &  & 71.9      & 59.74       & 38.93    & 45.7  &  & 78.23     & 59.18       & 39.16    & 43.93 \\
SEED=3                     & 0.5759   & 30              & 0.37833336     & 35               & 0.72457147      & 35                 & 0.5965714         & 73.03     & 60          & 33       & 70.39 &  & 72.35     & 59.75       & 34.26    & 68.44 &  & 72.07     & 60.23       & 36.28    & 66.12 &  & 72.4      & 59.79       & 37.03    & 63.89 &  & 72.63     & 59.36       & 37.94    & 60.47 &  & 72.46     & 59.66       & 37.83    & 57.59 &  & 72.68     & 59.45       & 37.94    & 54.65 &  & 73.51     & 60.42       & 38.22    & 51.26 &  & 75.14     & 60.74       & 38.09    & 48.79 &  & 72.04     & 59.85       & 37.91    & 44.89 &  & 77.69     & 59.24       & 37.88    & 42.76 \\
SEED=4                     & 0.5692   & 30              & 0.37333333     & 35               & 0.7211428       & 35                 & 0.5851429         & 72.65     & 58.98       & 41.5     & 69.83 &  & 72.02     & 59.09       & 37.08    & 67.93 &  & 71.8      & 58.43       & 36.66    & 65.35 &  & 71.85     & 57.9        & 36.9     & 62.94 &  & 71.84     & 57.92       & 38.08    & 59.66 &  & 72.11     & 58.51       & 37.33    & 56.92 &  & 71.8      & 58.29       & 37.78    & 53.96 &  & 73.56     & 58.58       & 38.18    & 50.69 &  & 73.95     & 58.21       & 38.03    & 47.83 &  & 71.57     & 56.33       & 37.92    & 44.05 &  & 78.27     & 56.55       & 37.7     & 42.32 \\
SEED=5                     & 0.5706   & 30              & 0.37133333     & 35               & 0.7254286       & 35                 & 0.58657146        & 73.27     & 58.82       & 35.83    & 69.91 &  & 72.25     & 58.6        & 34.43    & 67.76 &  & 72.14     & 58.16       & 35.41    & 65.3  &  & 72.38     & 57.81       & 37.39    & 63.2  &  & 72.46     & 57.9        & 38.48    & 59.96 &  & 72.54     & 58.66       & 37.13    & 57.06 &  & 72.51     & 58.59       & 37.42    & 54.16 &  & 74.51     & 58.83       & 38.06    & 50.95 &  & 76.27     & 58.65       & 37.73    & 48.17 &  & 73.5      & 57.5        & 37.6     & 44.38 &  & 78.78     & 56.12       & 37.4     & 42.04 \\
mean                       & 0.57314  & 30              & 0.375466676    & 35               & 0.725085734     & 35                 & 0.590628592       & 72.972    & 60.12       & 36.11    & 70.17 &  & 72.302    & 59.622      & 35.004   & 68.17 &  & 72.162    & 59.19       & 36.34    & 65.76 &  & 72.4      & 58.674      & 37.61    & 63.57 &  & 72.518    & 58.59       & 38.424   & 60.24 &  & 72.508    & 59.064      & 37.544   & 57.31 &  & 72.484    & 58.922      & 37.912   & 54.43 &  & 73.684    & 59.348      & 38.446   & 51.11 &  & 75.01     & 59.402      & 38.258   & 48.48 &  & 72.088    & 58.436      & 38.254   & 44.81 &  & 77.848    & 57.888      & 38.224   & 42.81 \\
standard   deviation       & 0.003042 & 0               & 0.00298702     & 0                & 0.002750921     & 0                  & 0.0047095         & 0.4023928 & 1.289437862 & 3.271185 & 0.508 &  & 0.5157228 & 0.900705279 & 1.494868 & 0.607 &  & 0.383888  & 0.916869674 & 1.054348 & 0.554 &  & 0.4364058 & 0.825911618 & 0.718575 & 0.501 &  & 0.4354538 & 0.655438784 & 0.419798 & 0.415 &  & 0.2777949 & 0.473001057 & 0.299132 & 0.304 &  & 0.4577445 & 0.481788335 & 0.332746 & 0.349 &  & 0.6158571 & 0.725651431 & 0.446296 & 0.368 &  & 0.8346556 & 1.009985148 & 0.488385 & 0.506 &  & 0.826541  & 1.509645654 & 0.621474 & 0.634 &  & 0.9627668 & 1.46856052  & 0.79365  & 0.729 \\
                           &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
iNaturalist   2018         &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
                           &          &                 &                &                  &                 &                    &                   & \multicolumn{4}{c}{forward3}               &  & \multicolumn{4}{c}{forward2}               &  & \multicolumn{4}{c}{uniform}                &  & \multicolumn{4}{c}{backward2}              &  & \multicolumn{4}{c}{backward3}              &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
Standard-training   regime & 0.749202 & 3224            & 0.75093055     & 842              & 0.733175        & 4076               & 0.75114495        & 72.76     & 74.96       & nan      & 74.16 &  & 72.88     & 75.27       & 74.72    & 74.86 &  & 73.32     & 75.11       & 75.09    & 74.92 &  & 72.68     & 75.38       & 74.67    & 75.04 &  & nan       & 75..2       & 75.18    & 75.14 &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
                           &          &                 &                &                  &                 &                    &                   &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       \\
long-training   regime     & 0.770531 & 3224            & 0.7664392      & 842              & 0.7486144       & 4076               & 0.77829576        & 74.6      & 77.77       & nan      & 76.38 &  & 74.6      & 77.88       & 76.95    & 77.21 &  & 74.86     & 77.83       & 76.64    & 77.05 &  & 75.89     & 78.28       & 75.91    & 77.11 &  & nan       & 78.12       & 76.38    & 77.08 &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &       &  &           &             &          &      
\end{tabular}
\end{table}

# 🌎 [CVPR 2023] BalPoE-CalibratedLT
by **Emanuel Sanchez Aimar, Arvi Jonnarth, Michael Felsberg, Marco Kuhlmann**

This repository contains the official Pytorch implementation of [Balanced Product of Calibrated Experts for Long-Tailed Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Aimar_Balanced_Product_of_Calibrated_Experts_for_Long-Tailed_Recognition_CVPR_2023_paper.pdf) at CVPR 2023.

🎬[Video](https://www.youtube.com/watch?v=H664_EQq2cs) | 💻[Slides](assets/CVPR2023-short-presentation.pdf) | 🔥[Poster](assets/BalPoE-CalibratedLT-poster.pdf) | [ArXiv preprint](https://arxiv.org/abs/2206.05260) 

## Method

<p align="center"> <img src='assets/balpoe_calibrated_framework.png' align="center"> </p>

## Abstract
Many real-world recognition problems are characterized by long-tailed label distributions. These distributions make representation learning highly challenging due to limited generalization over the tail classes. If the test distribution differs from the training distribution, e.g. uniform versus long-tailed, the problem of the distribution shift needs to be addressed. A recent line of work proposes learning multiple diverse experts to tackle this issue. Ensemble diversity is encouraged by various techniques, e.g. by specializing different experts in the head and the tail classes. In this work, we take an analytical approach and extend the notion of logit adjustment to ensembles to form a Balanced Product of Experts (BalPoE). BalPoE combines a family of experts with different test-time target distributions, generalizing several previous approaches. We show how to properly define these distributions and combine the experts in order to achieve unbiased predictions, by proving that the ensemble is Fisher-consistent for minimizing the balanced error. Our theoretical analysis shows that our balanced ensemble requires calibrated experts, which we achieve in practice using mixup. We conduct extensive experiments and our method obtains new state-of-the-art results on three long-tailed datasets: CIFAR-100-LT, ImageNet-LT, and iNaturalist-2018.

## Getting Started

### Prerequisites

BalPoE-CalibratedLT is built on pytorch and a handful of other open-source libraries.

To install the required packages, you can create a conda environment:

```sh
conda create --name balpoe python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Hardware requirements
4 GPUs with >= 24G GPU RAM are recommended (for large datasets). Otherwise, the model with more experts may not fit in, especially on datasets with more classes (the FC layers will be large). We do not support CPU training at the moment.

## Datasets
### Four benchmark datasets
* Please download these datasets and put them to the /data file.
* CIFAR-100 / CIFAR-10 will be downloaded automatically with the dataloader.
* iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).
* ImageNet-LT can be found at [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).

### Txt files
* We provide txt files for long-tailed recognition under multiple test distributions for ImageNet-LT and iNaturalist 2018. CIFAR-100 will be generated automatically with the code.
* For iNaturalist 2018, please unzip the iNaturalist_train.zip.
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_backward2.txt
│   ├── ImageNet_LT_backward5.txt
│   ├── ImageNet_LT_backward10.txt
│   ├── ImageNet_LT_backward25.txt
│   ├── ImageNet_LT_backward50.txt
│   ├── ImageNet_LT_forward2.txt
│   ├── ImageNet_LT_forward5.txt
│   ├── ImageNet_LT_forward10.txt
│   ├── ImageNet_LT_forward25.txt
│   ├── ImageNet_LT_forward50.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_uniform.txt
│   └── ImageNet_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_backward2.txt
    ├── iNaturalist18_backward3.txt
    ├── iNaturalist18_forward2.txt
    ├── iNaturalist18_forward3.txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```

## Usage

### CIFAR100-LT 
#### Training

* Important: to reproduce our main results, train five runs with SEED = {1,2,3,4,5} and compute mean and standard deviation over reported results.

* To train BalPoE with three experts on the standard-training regime, run this command:
```
python train.py -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --seed 1
```

* To train BalPoE with three experts on the long-training regime using AutoAugment, run:
```
python train.py -c configs/mixup/long_training/config_cifar100_ir100_bs-experts.json --seed 1
```

* One can change the imbalance ratio from 100 to 10/50 by changing the config file. Similar instructions for CIFAR10-LT.
* Optionally, to train our framework with more experts, e.g. 7 experts, run:
```
python train.py -c configs/mixup/long_training/config_cifar100_ir100_bs-experts.json --tau_list limits --num_experts 7 --seed 1
```


#### Evaluate
* To evaluate BalPoE on the uniform distribution, run:
``` 
python test.py -r checkpoint_path
```

where checkpoint_path should be of the form CHECKPOINT_DIR/checkpoint-epoch[LAST_EPOCH].pth, 
where LAST_EPOCH is 200 and 400 for standard and long training, respectively.

* To evaluate on diverse test class distributions, run:
``` 
python test_all_cifar.py -r checkpoint_path [--posthoc_bias_correction]
```

Optional: use --posthoc_bias_correction to adjust logits with known test prior.

### iNaturalist 2018
#### Training
* To train BalPoE with three experts on the standard-training regime, run this command:
```
python train.py -c configs/mixup/standard_training/config_iNaturalist_resnet50_bs-experts.json
``` 

* To train BalPoE with three experts on the long-training regime using RandAugment, run this command:
```
python train.py -c configs/mixup/long_training/config_iNaturalist_resnet50_bs-experts.json
```

#### Evaluate
* To evaluate BalPoE on the uniform distribution, run:
``` 
python test.py -r checkpoint_path
```

where checkpoint_path should be of the form CHECKPOINT_DIR/checkpoint-epoch[LAST_EPOCH].pth, 
where LAST_EPOCH is 100 and 400 for standard and long training, respectively.

* To evaluate on diverse test class distributions, run:
``` 
python test_all_inat.py -r checkpoint_path [--posthoc_bias_correction]
``` 

Optional: use --posthoc_bias_correction to adjust logits with known test prior.

### ImageNet-LT
#### Training
* To train BalPoE with three experts on the standard-training regime, run this command:
```
python train.py -c configs/mixup/standard_training/config_imagenet_lt_resnext50_bs-experts.json
```

* To train BalPoE with three experts on the long-training regime using RandAugment, run this command:
```
python train.py -c configs/mixup/long_training/config_imagenet_lt_resnext50_bs-experts.json
```

* Alternatively, train with ResNet50 backbone by using the corresponding config file.

#### Evaluate
* To evaluate BalPoE on the uniform distribution, run:
``` 
python test.py -r checkpoint_path
```
where checkpoint_path should be of the form CHECKPOINT_DIR/checkpoint-epoch[LAST_EPOCH].pth, 
where LAST_EPOCH is 180 and 400 for standard and long training, respectively.

* To evaluate the model on different test distributions, run:
``` 
python test_all_imagenet.py -r checkpoint_path
```

Optional: use run the following command to accomodate for a known test prior.
``` 
python test_all_imagenet.py -r checkpoint_path --posthoc_bias_correction
```

## Citation
If you find our work inspiring or use our codebase in your research, please cite our work.
```
@InProceedings{SanchezAimar2023BalPoE_CalibratedLT,
    author    = {Sanchez Aimar, Emanuel and Jonnarth, Arvi and Felsberg, Michael and Kuhlmann, Marco},
    title     = {Balanced Product of Calibrated Experts for Long-Tailed Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19967-19977}
}
```

## Acknowledgements

Our codebase is based on several open-source projects, particularly: 
- [SADE](https://github.com/Vanint/SADE-AgnosticLT) 
- [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition)
- [BagOfTricksLT](https://github.com/zhangyongshun/BagofTricks-LT)

This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). 
