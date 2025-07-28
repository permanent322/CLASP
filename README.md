# CLTSAD

### Create an virtual environment
```
conda create --name cltsad python=3.9 -y
conda activate  cltsad
pip install -r requirements.txt
```


### Download data

```
python download_data.py swat
python download_data.py smd
python download_data.py smap
python download_data.py psm
python download_data.py msl
```

### Train and Test

```
chmod +x run.sh
./run_alldata.sh
```