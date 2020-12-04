FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER 1000:1000

COPY . .

# time series analysis
# time series forcasting