from ai import train
from ai import convert_csv
from ai import financial_api

financial_api.download_everything()
convert_csv.main()
train.main()