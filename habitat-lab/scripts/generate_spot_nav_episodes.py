#!/usr/bin/env python3

import argparse
import glob
import gzip
import json
import multiprocessing
import os
from os import path as osp

import tqdm

import habitat
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)

parser = argparse.ArgumentParser()
parser.add_argument('config_yaml')
parser.add_argument('out_dir')
parser.add_argument('num_episodes', type=int)
parser.add_argument('glb_dir')
parser.add_argument('robot')
parser.add_argument('-t', '--ty', type=str)
parser.add_argument('-si', '--start-idx', type=int)
parser.add_argument('-ei', '--end-idx', type=int)
parser.add_argument('-v', '--val', action='store_true')
args = parser.parse_args()

# Task config yaml file. Needed for agent radius and height, not sure what else...
CONFIG_YAML = args.config_yaml

# Folder where json.gz files will be saved:
TRAIN_EP_DIR = args.out_dir

NUM_EPISODES_PER_SCENE = args.num_episodes

if args.ty == 'hm3d':
    VAL = [
        '00800-TEEsavR23oF', '00801-HaxA7YrQdEC', '00802-wcojb4TFT35', '00803-k1cupFYWXJ6', '00804-BHXhpBwSMLh', '00805-SUHsP6z2gcJ', '00806-tQ5s4ShP627', '00807-rsggHU7g7dh', '00808-y9hTuugGdiq', '00809-Qpor2mEya8F', '00810-CrMo8WxCyVb', '00811-7UrtFsADwob', '00812-mma8eWq3nNQ', '00813-svBbv1Pavdk', '00814-p53SfW6mjZe', '00815-h1zeeAwLh9Z', '00816-q3hn1WQ12rz', '00817-X4qjx5vquwH', '00818-rJhMRvNn4DS', '00819-6D36GQHuP8H', '00820-mL8ThkuaVTM', '00821-eF36g7L6Z9M', '00822-nrA1tAA17Yp', '00823-7MXmsvcQjpJ', '00824-Dd4bFSTQ8gi', '00825-jgPBycuV1Jq', '00826-BFRyYbPCCPE', '00827-BAbdmeyTvMZ', '00828-X7gTkoDHViv', '00829-QaLdnwvtxbs', '00830-5jp3fCRSRjc', '00831-yr17PDCnDDW', '00832-qyAac8rV8Zk', '00833-dHwjuKfkRUR', '00834-AWUFxHEyV3T', '00835-q3zU7Yy5E5s', '00836-7GAhQPFzMot', '00837-vBMLrTe4uLA', '00838-kJJyRFXVpx2', '00839-zt1RVoi7PcG', '00840-z9YwN9M8FpG', '00841-SByzJLxpRGn', '00842-hkr2MGpHD6B', '00843-DYehNKdT76V', '00844-q5QZSEeHe5g', '00845-c3WKCnkEdha', '00846-LNg5mXe1BDj', '00847-bCPU9suPUw9', '00848-ziup5kvtCCR', '00849-a8BtkwhxdRV', '00850-W7k2QWzBrFY', '00851-YRUkbU5xsYj', '00852-T6nG3E2Uui9', '00853-5cdEh9F2hJL', '00854-SiKqEZx7Ejt', '00855-c5eTyR3Rxyh', '00856-FnSn2KSrALj', '00857-L53DsRRk4Ch', '00858-cYkrGrCg2kB', '00859-3t8DB4Uzvkt', '00860-RJaJt8UjXav', '00861-GLAQ4DNUx5U', '00862-LT9Jq6dN3Ea', '00863-b28CWbpQvor', '00864-QHhQZWdMpGJ', '00865-fsQtJ8t3nTf', '00866-XNeHsjL6nBB', '00867-uLz9jNga3kC', '00868-vd3HHTEpmyA', '00869-MHPLjHsuG27', '00870-AYpsNQsWncn', '00871-VBzV5z6i1WS', '00872-F7EAMsdDASd', '00873-bxsVRursffK', '00874-uSKXQ5fFg6u', '00875-66seV3BWPoX', '00876-mv2HUxq3B53', '00877-4ok3usBNeis', '00878-XB4GS9ShBRE', '00879-7Ukhou1GxYi', '00880-Nfvxx8J5NCo', '00881-u1bkiGRVyu9', '00882-rXXL6twQiWc', '00883-u8ug2rtNARf', '00884-XMHNu9rRQ1y', '00885-dVW2D7TDctW', '00886-TPhiubUHKcP', '00887-hyFzGGJCSYs', '00888-hDBqLgydy1n', '00889-HMkoS756sz6', '00890-6s7QHgap2fW', '00891-cvZr5TUy5C5', '00892-bzCsHPLDztK', '00893-yZME6UR9dUN', '00894-HY1NcmCgn3n', '00895-auFeVz9Go4m', '00896-5hGmubEE6ZF', '00897-LEFTm3JecaC', '00898-8CRYizAb6yd', '00899-58NLZxWBSpk',
    ]

    TRAIN = [
        '00000-kfPV7w3FaU5', '00001-UVdNNRcVyV1', '00002-FxCkHAfgh7A', '00003-NtVbfPCkBFy', '00004-VqCaAuuoeWk', 
        '00005-yPKGKBCyYx8', '00006-HkseAnWCgqk', '00007-UQuchpekHRJ', '00008-VYnUX657cVo', '00009-vLpv2VX547B', 
        '00010-DBjEcHFg4oq', '00011-1W61QJVDBqe', '00012-kDgLKdMd5X8', '00013-sfbj7jspYWj', '00014-nYYcLpSzihC', 
        '00015-LPwS1aEGXBb', '00016-qk9eeNeR4vw', '00017-oEPjPNSPmzL', '00018-as8Y8AYx6yW', '00019-AfKhsVmG8L4', 
        '00020-XYyR54sxe6b', '00021-yQESfVcg18k', '00022-gmuS7Wgsbrx', '00023-zepmXAdrpjR', '00024-XNoaAZwsWKk', 
        '00025-ixTj1aTMup2', '00026-tzaZQQmUVXZ', '00027-cVppJowrUqs', '00028-xGnehmjiCSA', '00029-4wCTuaUNWEd', 
        '00030-LPEMkRVudUm', '00031-Wo6kuutE9i7', '00032-jTTGECZYKRA', '00033-oPj9qMxrDEa', '00034-6imZUJGRUq4', 
        '00035-3XYAD64HpDr', '00036-41FNXLAZZgC', '00037-oKFJo8jpzRW', '00038-aJg466zMSNt', '00039-ANmWrL7Kz7h', 
        '00040-ZB8o8rMmPdB', '00041-QKfBMSSy7Hy', '00042-qDjhFcNqFPi', '00043-Jfyvj3xn2aJ', '00044-kuxpR2xHUBa', 
        '00045-mt5PhMmTk5m', '00046-UQ5EhY5wve1', '00047-u9LiqMn6kA6', '00048-aKwZ7VJDAjr', '00049-e3YKRHQRPNe', 
        '00050-wrq3kiEU4VR', '00051-CETmJJqkhcK', '00052-H1D2FZ8TAv1', '00053-kAMF2R7PCqX', '00054-6BReaxZUoMg', 
        '00055-HxmXPBbFCkH', '00056-kdw2Uapns3b', '00057-1UnKg1rAb8A', '00058-1hovphK64XQ', '00059-kJxT5qssH4H', 
        '00060-TgWKHxhJAng', '00061-duthTPisf28', '00062-ACZZiU6BXLz', '00063-KsD3yx9nZCv', '00064-gQgtJ9Stk5s', 
        '00065-kZhZfAhdnNN', '00066-nHHVbEyHX3t', '00067-osQy15y8EVT', '00068-812QqCky3T7', '00069-Y8Y6ukxGMvn', 
        '00070-w7QyjJ3H9Bp', '00071-GCEb4nmNi7j', '00072-t8wCA6Qe8uT', '00073-bCFcvb4zc3N', '00074-kHvNo8x6Qoe', 
        '00075-u3zrj4Nojev', '00076-fJ1GEE6PdHD', '00077-z9VLaZqCsW5', '00078-nJTPfwbAj4S', '00079-2ihrkjrbHVf', 
        '00080-gPf8XzHACGn', '00081-5biL7VEkByM', '00082-y4YiUQwvWGH', '00083-16tymPtM7uS', '00084-CtZLhCbWFm7', 
        '00085-KWoYKrff5L4', '00086-315QujoX279', '00087-YY8rqV6L6rf', '00088-z9w4aD7JsiQ', '00089-Am8wmPcBmtN', 
        '00090-C8VQQtzUoqV', '00091-SA759ShSs44', '00092-1mCzDx3EMom', '00093-HneuS3CYDFG', '00094-WT4QWwXrMzs', 
        '00095-HsYeeztxPG6', '00096-6HRFAUDqpTb', '00097-rPwrKEnR3fk', '00098-MVStfLiYYQv', '00099-226REUyJh2K', 
        '00100-y3K5dmhuukt', '00101-n8AnEznQQpv', '00102-r77mpaAYUEc', '00103-gUqgeUmUagL', '00104-KJxdMPgweZG', 
        '00105-xWvSkKiWQpC', '00106-ZVScmfktNQ1', '00107-Y6WjWkVEUks', '00108-oStKKWkQ1id', '00109-GTV2Y73Sn5t', 
        '00110-ikChNYDHtRf', '00111-AMEM2eWycTq', '00112-r38SGhq8aJr', '00113-3goH1WRaCYC', '00114-Coer9RdivP7', 
        '00115-NBWrHFXBF5p', '00116-xp4FyfQ6Wr5', '00117-2NwLiyeKcrK', '00118-F5j7ZLfMm1n', '00119-k17yptqNRAn', 
        '00120-eAUmfFLZDR3', '00121-D2PqRE5ZvyQ', '00122-QDtpZSqaeyW', '00123-C3ifY177Ldq', '00124-GZRLndzSrdn', 
        '00125-6acdNdTjNbr', '00126-dioA6agn1cP', '00127-EN7GiDgxdQ2', '00128-pAjDzi9kWjE', '00129-nd6Vw5SHCoy', 
        '00130-MjzBosUX3WW', '00131-bZsfeA9uRk7', '00132-sWAdxSLVPQC', '00133-P3kZKzwnEbM', '00134-Mre6deDcPCP', 
        '00135-HeSYRw7eMtG', '00136-z8SrvZ4eyqV', '00137-2tqbn5VLQoq', '00138-jAzZDvf6i67', '00139-Dj6sJyU7qmd', 
        '00140-qWb4MVxqCW7', '00141-iigzG1rtanx', '00142-DwDDvGo9QdA', '00143-5Kw4nGdqYtS', '00144-7CXbc73tDRf', 
        '00145-U8F9SkAsqbJ', '00146-BLhX6Do8f1t', '00147-TZ2jsvNG2nt', '00148-csHQLLFPE3g', '00149-UuwwmrTsfBN', 
        '00150-LcAd9dhvVwh', '00151-D8bT1ambLFc', '00152-EgzWe8N3jZG', '00153-28FFMGySc6D', '00154-2dZ1Jivh5if', 
        '00155-iLDo95ZbDJq', '00156-iQPq34e8hJX', '00157-3PiKdwyfEkX', '00158-YV9M9gZG3YJ', '00159-DfUQLukPMPc', 
        '00160-7dmR22gwQpH', '00161-uFCiZVVks57', '00162-1sM6KvYg3J5', '00163-uXh7Yr7x12L', '00164-XfUxBGTFQQb', 
        '00165-N8oi63yAP2b', '00166-RaYrxWt5pR1', '00167-yogvKWUrdnw', '00168-bHKTDQFJxTw', '00169-3KZbo846fxq', 
        '00170-S3r45BMWy6H', '00171-mWqBmEyXcXN', '00172-bB6nKqfsb1z', '00173-qZ4B7U6XE5Y', '00174-BqLwEyiLbza', 
        '00175-nW7z5USWzWo', '00176-wcJYziD5pmF', '00177-VSxVP19Cdyw', '00178-yqNxxJnA3iL', '00179-MVVzj944atG', 
        '00180-sCCThjhioJC', '00181-qkkRnWghK8e', '00182-qWP3MMQM3eJ', '00183-kEL17iFsVbw', '00184-nzuiinFMXvf', 
        '00185-NjyeoK5BLx3', '00186-7UdY7HiDnUi', '00187-GCCrNuhZ9WY', '00188-dQrLTxHvLXU', '00189-KHhgcNqsc9h', 
        '00190-NkvRYHk72vA', '00191-dKySjDYsya1', '00192-MLVm7dZk7dp', '00193-wwX4MFiTTrt', '00194-JY8e73x9ubE', 
        '00195-enfahKs8XHw', '00196-3Y14etT7365', '00197-x4LVLSsYWcV', '00198-eZrc5fLTCmi', '00199-uHnM1oqv2JL', 
        '00200-A9yB3w3UxXV', '00201-k7vRbGpz44m', '00202-yVbpFay8gTU', '00203-VoVGtfYrpuQ', '00204-gxttMtT5ZGK', 
        '00205-NEVASPhcrxR', '00206-uhkqDVMtEnn', '00207-FRQ75PjD278', '00208-SQqGpSHzfSr', '00209-C5RbHBQ76DE', 
        '00210-j2EJhFEQGCL', '00211-hmRxh2mmzNC', '00212-bAdy4hKf1a1', '00213-mkvHBa3mEEk', '00214-WeyCwVzL53K', 
        '00215-zWydhyFhvcj', '00216-6EMViBCA2N7', '00217-qz3829g1Lzf', '00218-fQHGxvurx9L', '00219-P6ajptD9tRP', 
        '00220-KAzjXJvZtR3', '00221-zJEEFaNaRbB', '00222-g8Xrdbe9fir', '00223-wQN24R38a9N', '00224-8m2Sk2sC3DR', 
        '00225-EbiLVt7CHc1', '00226-8mXffaQTtmP', '00227-bvdRzJBgJyg', '00228-d5faFfQRphr', '00229-cHumXFzhHUR', 
        '00230-TNx8nti6GNi', '00231-4h4JxvG3cip', '00232-XNiSi1YgPRR', '00233-PuFu1zFVc4k', '00234-nACV8wLu1u5', 
        '00235-DWsM2ficpne', '00236-FcUgaJv6JHA', '00237-BW1f54ZNVW6', '00238-j6fHrce9pHR', '00239-b3CuYvwpzZv', 
        '00240-q9CAdKfvar2', '00241-h6nwVLpAKQz', '00242-JHHVv6QZJMm', '00243-BfzKZxFShtq', '00244-E64sjs3Dyfd', 
        '00245-741Fdj7NLF9', '00246-mscxX4KEBcB', '00247-uvveHdpUZis', '00248-PM558qFsyi8', '00249-PYHRaSotJNh', 
        '00250-U3oQjwTuMX8', '00251-wsAYBFtQaL7', '00252-kGVKV9k9DCT', '00253-HjxjHvpdeoM', '00254-YMNvYDhK8mB', 
        '00255-NGyoyh91xXJ', '00256-92vYG1q49FY', '00257-j2DKmTV5TPV', '00258-2Pc8W48bu21', '00259-4J8N2Ah1a6o', 
        '00260-ECStCRoCNWM', '00261-fK2vEV32Lag', '00262-1xGrZPxG1Hz', '00263-GGBvSFddQgs', '00264-GfF9TQ34x37', 
        '00265-3Ao63EY7J83', '00266-67ADtrTrBK2', '00267-gQ3xxshDiCz', '00268-mHJxL9jnCox', '00269-JNiWU5TZLtt', 
        '00270-bDTsgcSK5Qr', '00271-cjLuWviyDEo', '00272-kA2nG18hCAr', '00273-8DDKELpgD99', '00274-NcK5aACg44h', 
        '00275-4dbCzNN5L5t', '00276-HLBJGGyicLV', '00277-RHrFkyC59tf', '00278-dcd823nTKH9', '00279-YmEfzspXX5h', 
        '00280-FgswoxWb3uN', '00281-mj5sKX44BmS', '00282-8nSVqHLMRwQ', '00283-MPPDV4Gvybr', '00284-4GfZ9TTZUwL', 
        '00285-QKGMrurUVbk', '00286-UbsJXeCkJBA', '00287-SBHLgvFTVMZ', '00288-27cQLjQ5CjV', '00289-wz9FcGhrndc', 
        '00290-6HMiy15cxis', '00291-Umx6CdjZfvy', '00292-reHtN7VMWkg', '00293-pEeGyoYCEa1', '00294-PPTLa8SkUfo', 
        '00295-LQy8D2nmZ4x', '00296-sLwz8nKD3wF', '00297-jGdNyKqGZJw', '00298-by8SK9u18S8', '00299-bdp1XNEdvmW', 
        '00300-GcfUJ79xCZc', '00301-JiHGQpwKUvd', '00302-JFgrz9MNz4b', '00303-ghWQ5kHV97i', '00304-X6Pct1msZv5', 
        '00305-W3J8ruZTQic', '00306-Y4L8fjz2yH7', '00307-vDfkYo5VqEQ', '00308-gmLDom6XSo7', '00309-VKmpsujnc5t', 
        '00310-WnvnMQh4eEa', '00311-mHXUEKEV6gR', '00312-UrFKpVJpvHi', '00313-PE6kVEtrxtj', '00314-NwG7cpZnRZb', 
        '00315-We1N7vBtyGm', '00316-LqsTKpxKVP2', '00317-P8XJUpcAUkf', '00318-6qJyEsZNuey', '00319-sjH1uaR68XQ', 
        '00320-nicaPonCxvC', '00321-JWWJBQWHv64', '00322-7Xp37y8DpSv', '00323-yHLr6bvWsVm', '00324-DoSbsoo4EAg', 
        '00325-bMvM1KL4WsR', '00326-u9rPN5cHWBg', '00327-xgLmjqzoAzF', '00328-DACaFbApXUe', '00329-asrq4PFvdvF', 
        '00330-WhNyDTnd9g5', '00331-gGMMut83nsX', '00332-cWfRoQnzNiM', '00333-LLecyBe5Eq2', '00334-hkmLgL6jrP8', 
        '00335-janiYDpzM9j', '00336-uc8QkFS11Hj', '00337-CFVBbU9Rsyb', '00338-rK4jPRTUw15', '00339-T7nCRmufFNR', 
        '00340-GtqoUWABJ11', '00341-tYvWp85L81G', '00342-C6JvMamYTRg', '00343-5graSmdK3Bj', '00344-PaQrTquNd2v', 
        '00345-8EqKbkhqE4R', '00346-mQFC1yx29MM', '00347-bwoBmU23M2N', '00348-37c5w29pYm3', '00349-p32JzpQyhPk', 
        '00350-amdKHm5GAM1', '00351-QxfX5te1gFu', '00352-zmZvNTCxMZE', '00353-8qbZhbTc1wX', '00354-NpCFg9NdUgL', 
        '00355-Qkm4CooNoPi', '00356-RYzud5W7ZnC', '00357-TzQLNfWugiZ', '00358-Wjvsh6jnVsR', '00359-aTf5zsbjZMb', 
        '00360-yA3RqPqMrGE', '00361-uXU4dVyyvWa', '00362-o94q92w5PK5', '00363-DS3nSAaa3Nr', '00364-tEafuWwhhwr', 
        '00365-tvvDpjzFJGe', '00366-fxbzYAGkrtm', '00367-RHdkyzXFp1k', '00368-k9UfRPqLm3j', '00369-FgXPKxNp5kK', 
        '00370-RcuYAHzrjK7', '00371-6AGcGQf2wof', '00372-GMwtBqNLGBs', '00373-WPzCkWEorzk', '00374-j2eqyxdYAFW', 
        '00375-kubNyvKJBUX', '00376-wxixLWuvLjd', '00377-Fgtk7tL8R9Y', '00378-DqJKU7YU7dA', '00379-58fkJMgLopt', '00380-fdfRpE6Cfsr', '00381-pmZMJfFd3Jy', '00382-S7uMvxjBVZq', '00383-yTgw14aa5ha', '00384-ceJTwFNjqCt', '00385-PB8eQHRTRRK', '00386-b3WpMbPFB6q', '00387-6vJMULqvYe8', '00388-pcpn6mFqFCg', '00389-d6bYiL1d9Fh', '00390-c9DeZf2fcDf', '00391-3UDjdrwcqMb', '00392-M8PMwoYQTUV', '00393-dD37vuCa4FE', '00394-H7bBanejcc6', '00395-giViJCyCH2C', '00396-wuKkTq5GJbi', '00397-ZNanfzgCdm3', '00398-Vnb6uKtzQCU', '00399-XokRUNE3gB1', '00400-KcHdFEzySGq', '00401-H8rQCnvBgo6', '00402-zR6kPe1PsyS', '00403-t3t9ofFLcFU', '00404-QN2dRqwd84J', '00405-q28T9C3q2dv', '00406-n2Tt2eJdqnT', '00407-NPHxDe6VeCc', '00408-RamZzGBBPbT', '00409-rxGLNxH6eoJ', '00410-v7DzfFFEpsD', '00411-o4tckGBtaxz', '00412-mDPCxA7W1WN', '00413-YM4nG4pSAEJ', '00414-77mMEyxhs44', '00415-rBmEe6ab5VP', '00416-zCMdfYaW9iF', '00417-nGhNxKrgBPb', '00418-t6tH2pNA9X5', '00419-fbUcgfPMBDr', '00420-R6Byftz8wRN', '00421-gamLwhSzHci', '00422-8wJuSPJ9FXG', '00423-bEdki9cbHDG', '00424-Hsk3jDzNySy', '00425-SSbSnMigayt', '00426-X9fRPGxw1jS', '00427-P3hmFK6Ejnf', '00428-vPoFkhqsJaf', '00429-UZ5rYGiwQgW', '00430-P8L1328HrLi', '00431-rp42RknmRV8', '00432-vCfHLVSyL21', '00433-uzH9yHazm9t', '00434-L5QEsaVqwrY', '00435-rmDFTEWfNcz', '00436-TGVJHgmMGzi', '00437-S3BfyR31Wc9', '00438-XvJjCZv6SYp', '00439-mjvN6RDLsPm', '00440-wPLokgvCnuk', '00441-4MRLu1yET6a', '00442-tJ7XVoEN82a', '00443-oz1yTAGPXkh', '00444-sX9xad6ULKc', '00445-H81QMurNRM8', '00446-tL6i2PtktSh', '00447-5RtSdesLuHt', '00448-tAQTHnJ7n72', '00449-JjsvQEqRxGS', '00450-LGGnLUDPz37', '00451-FYYpmNC4gAd', '00452-9Ckja165ren', '00453-6ySDHVkso9e', '00454-vjMVcmpC1hC', '00455-k7mcxHG65Wh', '00456-T48WJA2vtru', '00457-7PZPFHR3oJc', '00458-fFx6oC7EVp7', '00459-Ze6tkhg7Wvc', '00460-LViDMxZp4ZN', '00461-AiY2reLtjYJ', '00462-5m6t1y5EvsT', '00463-URjpCob8MGw', '00464-NfkadBDgBJV', '00465-nvjM2xMma91', '00466-xAHnY3QzFUN', '00467-mggziYKSc6S', '00468-iTm2PKHUcTJ', '00469-ochRmQAHtkF', '00470-udze1CSof5C', '00471-f93a9wrxRjG', '00472-Xuky7E5df6A', '00473-xccdSFAEPau', '00474-v46TaF2rxHK', '00475-g7hUFVNac26', '00476-NtnvZSMK3en', '00477-r8b8sRuxdt2', '00478-QDvRVeWFCjM', '00479-p4ZPcGtk6Ex', '00480-RrfVebebfWf', '00481-5jLhtVmWd5F', '00482-EQSguCqe5Rk', '00483-boHtwWDWtXh', '00484-fc7RfUCN5mY', '00485-yX54kr5c5g9', '00486-WypGcNbCdsH', '00487-erXNfWVjqZ8', '00488-F8PSGjTiv61', '00489-boJrpwDQtu9', '00490-BUFVGDCQNGb', '00491-Ty7djLDxQu3', '00492-panm7DRsmDn', '00493-pUneSGJDrvY', '00494-5737gQA9p2T', '00495-CQWES1bawee', '00496-eWKqgQbVZow', '00497-AeBfp3hTadB', '00498-ij6Fizhrr6c', '00499-q6tn1ZjSsG4', '00500-fKP4sxcoxpL', '00501-N7YVmJQ8sAu', '00502-nMeXfQU4PMS', '00503-aqx6EomMgTf', '00504-frThKkhTwFT', '00505-ZwnLFNzxASM', '00506-QVAA6zecMHu', '00507-RfNGMBdVbAZ', '00508-4vwGX7U38Ux', '00509-gDDiZeyaVc2', '00510-JSgMy8tTACD', '00511-8uSpPmctPXC', '00512-WZDzPCybQvS', '00513-9oGV6Y9nNqB', '00514-9DnDAhJ7qcj', '00515-krsjseyn6fd', '00516-ytXtEYghmvL', '00517-jG9pucmJVBZ', '00518-o1F5JVHc6mb', '00519-aNri5Gh1ZTE', '00520-LpFSk2s7me6', '00521-1SedVoP7zLu', '00522-XYQdAu1qsK9', '00523-PK83UqrXjd3', '00524-hbr5fTQhAAa', '00525-iKFn6fzyRqs', '00526-PFLHHbjscNN', '00527-HPrcqBkKzuy', '00528-BCWtGkh8CHv', '00529-W9YAR9qcuvN', '00530-C7xtw9uhYFn', '00531-TziyvKgzdAs', '00532-4euAVjDbkz9', '00533-wDDRygUCLMm', '00534-DBBESbk4Y3k', '00535-XKqDR74W1JU', '00536-QFjJExB2jgE', '00537-oahi4u45xMf', '00538-3CBBjsNkhqW', '00539-zUG6FL9TYeR', '00540-6nvpJEZ8ox5', '00541-FnDDfrBZPhh', '00542-12e6joG2B2T', '00543-1k479icNeHW', '00544-t85xiAt9pao', '00545-C2hbxeJWmvX', '00546-nS8T59Aw3sf', '00547-9h5JJxM6E5S', '00548-otTm4oTrHvc', '00549-TQSiMZJawkS', '00550-k3ohRuM6bso', '00551-zwzTbNq7xoW', '00552-eip6PNoeCPr', '00553-qAwGYe2GoZp', '00554-rihBP3nC6p4', '00555-Y4idBN66BqG', '00556-saBtfCeVoJ4', '00557-fRZhp6vWGw7', '00558-NieWWMV6tE4', '00559-53jtKd53a1X', '00560-gjhYih4upQ9', '00561-VhissfC8ggN', '00562-kCHmLFfMDuE', '00563-Bnq6SeZGL5b', '00564-hXHUtviUKBu', '00565-xc2kFoo9nbw', '00566-qmvPLqLAgvC', '00567-KjZrPggnHm8', '00568-suQdyWFG8g9', '00569-YJDUB7hWg9h', '00570-kyoZhaD9HuW', '00571-w5YEujJKsiy', '00572-dNASL765WSN', '00573-1zDbEdygBeW', '00574-bXu6SSWkJY8', '00575-pMntW4YkvvB', '00576-adgwjGh4NQK', '00577-kdFEfVoT1WE', '00578-dTzYwo8Hppu', '00579-9hJwm8k7Gka', '00580-xcTV5UHYHFV', '00581-WpVxtsP4xxA', '00582-TYDavTf8oyy', '00583-1wypxmRjuUR', '00584-ZxkSUELrWtQ', '00585-CxxHb5C8ZsP', '00586-qSom26FpYzR', '00587-JUANmB8jduD', '00588-YWy9hV7RfQB', '00589-NS5SeGunqiK', '00590-qgZhhx1MpTi', '00591-JptJPosx1Z6', '00592-CthA7sQNTPK', '00593-m17UDpW3tHm', '00594-1sPp3Wz8TCB', '00595-TBdN234fGEb', '00596-9SpHCfHaNiG', '00597-D8aaq3PH6dG', '00598-mt9H8KcxRKD', '00599-TiWanpmC63V', '00600-ENhuWpDE5EB', '00601-PjnDyQJJ3eM', '00602-XRHpoTZjtj7', '00603-Xfhi9GYbhqD', '00604-W4r5JssudHR', '00605-T22dejNjHK7', '00606-W16Bm4ysK8v', '00607-JXdzHne1mRo', '00608-j2Nms3h9XJv', '00609-x1pTUWx9DPr', '00610-1EiJpeRNEs1', '00611-PXAfUkZGMdU', '00612-GsQBY83r3hb', '00613-s19Uyn7AWwv', '00614-ki6Cu76pWzF', '00615-PUNuHY5M7MS', '00616-zhzot8MvSjF', '00617-AENiMBDjVFb', '00618-T4G9hTR5WSv', '00619-R9fYpvCUkV7', '00620-AUkcTmUs8mw', '00621-SAZ4gvMfxm1', '00622-bxwHR9ipFG8', '00623-g7sKCMRfgUS', '00624-ooq3SnvC79d', '00625-VWczCD1Hbus', '00626-XiJhRLvpKpX', '00627-b2e31HFFizw', '00628-tjs8mFdJ7YN', '00629-MHYu4LWb6qP', '00630-SrBPiU6LKxL', '00631-v3tsKAPVLJS', '00632-kXAEFtUBNFZ', '00633-dDyovSFuViJ', '00634-A1jHexSJuAW', '00635-drvaU627cQh', '00636-FXuXGH9YQTW', '00637-iNpfPhK1sRz', '00638-iePHCSf119p', '00639-AwL2QGztLwV', '00640-qvNra81N8BU', '00641-F8Rw6EWpPRT', '00642-erfqyP5V4u6', '00643-ggNAcMh8JPT', '00644-njMGKG4iwRK', '00645-rrjjmoZhZCo', '00646-UfhK7KNBg5u', '00647-qpcpnP8TosR', '00648-yCCyNxubYcL', '00649-RiwBKy2YdQ7', '00650-AqGu9nUng3L', '00651-kQVPtf7ACRw', '00652-BcZUZQ9t4Fe', '00653-4RuxhXRmb3V', '00654-VZy9kKQJcUF', '00655-q33GehreMrX', '00656-7GvCP12M9fi', '00657-TSJmdttd2GV', '00658-a3JCmxobR99', '00659-ZVPMj4YoZtK', '00660-Pmv1pdeirDT', '00661-SSwbmq72C21', '00662-aRKASs4e8j1', '00663-1Rg1SS1dRpG', '00664-u5atqC7vRCY', '00665-NBuk4gePdJm', '00666-GNGYKt8XrjF', '00667-5K2dTSVihN7', '00668-6YtDG3FhNvx', '00669-DNWbUAJYsPy', '00670-mDdyQ6azhVD', '00671-knPYW3fibqY', '00672-4L4peQsMgfR', '00673-WpAGGyZFqQj', '00674-y3dZtLZCUvN', '00675-c6TFyURFrL4', '00676-J5SkB2o1ckv', '00677-CnU5RD6PB3E', '00678-aCtdWA5n56Z', '00679-wFCLkVy4n9U', '00680-YmWinf3mhb5', '00681-QE33dvXyf1U', '00682-6r9GfBG7u1g', '00683-KCvzhHEhdwB', '00684-D5dEbkUphhr', '00685-ENiCjXWB6aQ', '00686-LU4A39yR8gc', '00687-tj8ngv3woJ3', '00688-SgkmkWjjmDJ', '00689-d88Sc1udFcZ', '00690-p6RF8AUer2e', '00691-MyxM6trMBUH', '00692-APXAdV48nKT', '00693-adddVdvEXUK', '00694-Lva3QmSMsTr', '00695-BJovXQkqbC3', '00696-DsEJeNPcZtE', '00697-5vUupbRRdyH', '00698-SrHVAbHUpUX', '00699-gyK27yu7CP4', '00700-aosjAwX5Lnq', '00701-tpxKD3awofe', '00702-wCqnXzoru3X', '00703-xvDx98avcwd', '00704-qnKYFQsjnHf', '00705-2XVvKEDd54w', '00706-YHmAkqgwe2p', '00707-XVSZJAtHKdi', '00708-eUJx9a4u63E', '00709-8LLjiNrWzJ9', '00710-DGXRxHddGAW', '00711-tTcuEfoAQXv', '00712-HZ2iMMBsBQ9', '00713-kxWk8ZMDE1N', '00714-Nf3aGQTDAA1', '00715-kjUg7BaQF1C', '00716-VaEwVD182FS', '00717-wtaQdtXzYtD', '00718-ASKXmHbw68X', '00719-CKbwkKufMWM', '00720-8B43pG641ff', '00721-YGc1h9nNrJP', '00722-EU6QPFpqdoU', '00723-hWDDQnSDMXb', '00724-H77MhktmmAF', '00725-5uXtMs57HmZ', '00726-sNikFfBW8zM', '00727-WEDXu8bWRkq', '00728-rWHyWNc6ZbZ', '00729-PyZonHqd5gy', '00730-UYrgg12a7QN', '00731-oXzJVhUhmYe', '00732-Z2DQddYp1fn', '00733-GtM3JtRvvvR', '00734-4dtmWCb8CDk', '00735-ypcVfePF8TG', '00736-h5VYFcePkbn', '00737-ZhXWtW1gd6c', '00738-GPyDUnjwZQy', '00739-MfkErJj6CHF', '00740-cFqWyQ4Y9hT', '00741-w8GiikYuFRk', '00742-aYhkzj2fEhP', '00743-31DHHWieDMS', '00744-1S7LAXRdDqK', '00745-yX5efd48dLf', '00746-RTV2n6fXB2w', '00747-ZKqtodH1qpa', '00748-8QtyGUUtacf', '00749-w3ZK3Wxvidz', '00750-E1NrAhMoqvB', '00751-DZsJKHoqEYg', '00752-vKuDtqxh7YQ', '00753-hTTnuAeSN6d', '00754-EqZacbtdApE', '00755-UAGeBzZJgkU', '00756-1K7P6ZQS4VM', '00757-LVgQNuK8vtv', '00758-HfMobPm86Xn', '00759-8iCxzGNmp4g', '00760-5Poh4Qz68hd', '00761-s7kPJndncRy', '00762-33ypawbKCQf', '00763-pVnwDTdMD3h', '00764-rzzVnFnBLtg', '00765-AuGMayXVFkc', '00766-XxbS57Z6PDU', '00767-Yr35Q49vqwV', '00768-Gg8W275oKZE', '00769-F1Vhvu3osn6', '00770-NBg5UqG3di3', '00771-8oSQng53cGV', '00772-zJ3fVx3BZYR', '00773-9K1WbyTZ456', '00774-S3YyrKoJ7k6', '00775-3z5dc2yzyCb', '00776-qxwfVS8MQ67', '00777-N17ddiDvJr9', '00778-vj4rZPfVjBQ', '00779-3YSDRj9kTU7', '00780-3iZkJUc7KhX', '00781-6TPCFES8fhh', '00782-oQPVc6vwgaq', '00783-J9adB1bm54A', '00784-pYGGNqSbHp1', '00785-AdNTcRg3THp', '00786-E9hxHD5h4FY', '00787-S9M7ybC5ZHu', '00788-BEuB32yj7Fb', '00789-UAByLdpaokx', '00790-qQgcM8T4hiD', '00791-JPMDv7zL4bF', '00792-4tdJ3qe1x7P', '00793-NRsmXFcVTbN', '00794-WRphMcFxfhe', '00795-awcRF7AZnJu', '00796-m49MsVC7BwA', '00797-99ML7CGPqsQ', '00798-bpqKLXHnxei', '00799-deNrXzuSss5'
    ]
elif args.ty == 'gibson':
    # 14 scenes
    VAL = [
        'Cantwell', 'Denmark', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Greigsville', 'Mosquito', 'Pablo',
        'Ribera', 'Sands', 'Scioto', 'Sisters', 'Swormville'
    ] 

    # 72 scenes
    TRAIN = [
        'Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw',
        'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Capistrano',
        'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Dryville', 'Dunmor', 'Eagerville', 'Goffs',
        'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic',
        'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Nemacolin', 'Nicut', 'Nimmons',
        'Nuevo', 'Oyens', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas',
        'Reyno', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sasakwa', 'Sawpit',
        'Seward', 'Shelbiana', 'Silas', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill',
        'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Woonsocket',
    ]

split = VAL if args.val else TRAIN
print('SPLIT: ', split)
if args.ty == 'hm3d':
    split = split[args.start_idx:args.end_idx+1]

    SCENES = []
    for x in split:
        SCENES.append(glob.glob(osp.abspath(osp.join(args.glb_dir, x, '*.glb')))[1])
elif args.ty == 'gibson':
    split = split[args.start_idx:args.end_idx+1]
    # List of paths to glb files
    SCENES = list(filter(
        lambda glb_path: any(list(
            map(
                lambda scene: scene in osp.basename(glb_path),
                split,
            )
        )),
        glob.glob(osp.abspath(osp.join(args.glb_dir, '*.glb'))),
    ))
SPLIT = split
print('SCENES: ', split)

# Pass absolute path to the scene glb
def _generate_fn(scene):
    scene_key = osp.basename(scene)
    print('SCENE KEY: ', scene_key)
    if args.ty == 'hm3d':
        for name in SPLIT:
            if scene_key.split('.')[0] in name:
                full_scene_key = name
    split = 'val' if args.val else 'train'      

    out_file = osp.join(
        TRAIN_EP_DIR, scene_key.split(".")[0]+".json.gz"
    )

    if osp.isfile(out_file):
        return

    ## radius, not diameter 
    if args.robot == 'A1':
        radius = 0.4/2
    elif args.robot == 'AlienGo':
        radius = 0.44/2
    elif args.robot == 'Locobot':
        radius = 0.45/2
    elif args.robot == 'Spot':
        radius = 0.6/2

    cfg = habitat.get_config(CONFIG_YAML)
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.RADIUS = radius 
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE,
            is_gen_shortest_path=False,
            shortest_path_max_steps= 150,
            geodesic_to_euclid_min_ratio=1.1,
            robot=args.robot
        )
    )
    print('scene: ', scene)
    for ep in dset.episodes:
        if args.ty == 'hm3d':
            ep.scene_id = osp.join('hm3d', split, full_scene_key, scene_key)
        elif args.ty == 'gibson':
            ep.scene_id = osp.join('gibson', scene_key)
    print('OUT FILE: ', out_file, len(dset.episodes))
    if len(dset.episodes) > 0:
        with gzip.open(out_file, "wt") as f:
            f.write(dset.to_json())

def generate_gibson_large_dataset():
    scenes = SCENES

    with multiprocessing.Pool(20) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()

if __name__ == "__main__":
    # print('\n'.join(SCENES))
    generate_gibson_large_dataset()