# Copy Results table for all material codes:
cp /home/vadcardm/TRAVAIL/DATA_ANALYSIS/Results.csv /home/elamrii/TRAVAIL/SI/Results.csv
chmod 777 /home/elamrii/TRAVAIL/SI/Results.csv

# Filter the results table on the material codes that make up the DF31 valve
gawk -F";" 'BEGIN{OFS=";"}{if($1=="10011988" || $1=="10011989" || $1=="10012074" || $1=="10012076" || $1=="10012153" || $1=="10031380" || $1=="10034501" || $1=="10039364" || $1=="10043390" || $1=="10043642" || $1=="10043660" || $1=="10047964" || $1=="10054944" || $1=="10054945" || $1=="10055021" || $1=="10055073" || $1=="10080949" || $1=="12162700" || $1=="12236310" || $1=="12290064" || $1=="12290479" || $1=="12290480" || $1=="12290481" || $1=="12312500" || $1=="12313900" || $1=="12315900" || $1=="12801295" || $1=="12801378" || $1=="12801379" || $1=="12802283" || $1=="12805284" || $1=="31099628"){print $0}}' /home/elamrii/TRAVAIL/SI/Results.csv > /home/elamrii/TRAVAIL/Valve_Project/data/Results.csv
