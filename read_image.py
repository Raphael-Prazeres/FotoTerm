import cv2
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData
from time import sleep
from tqdm import tqdm
from tkinter import *
# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)
from tkinter import filedialog

class FotoTerm:
    """Classe com métodos para o processamento de imagem
    """
    h = 780 # altura da imagem
    w = 540 # largura da imagem
    r = -2.5 # ângulo de rotação
    
    def __init__(self,altura:int,largura:int,angulo:float):
        """Inicializador da classe

        Args:
            altura (int): Altura da imagem em número de pixels.
            largura (int): Largura da imagem em número de pixels.
            angulo (float): Ângulo para rotacionar a imagem.
        """
        self.h = altura
        self.w = largura
        self.r = angulo
        
    
    def readImagem(self,caminho):
        """Faz o processamento da imagem retornado as retas para leitura dos sensores

        Args:
            caminho (str): caminho do arquivo de imagem

        Returns:
            DataFrame: Um dataFrame pandas com os pontos iniciais e finais das retas encontradas na imagem
        """
        # Read image
        image = cv2.imread(caminho)
        # Shape of image in terms of pixels.
        (rows, cols) = image.shape[:2]
        # getRotationMatrix2D creates a matrix needed for transformation.
        # We want matrix for rotation w.r.t center to -2.5 degree without scaling.
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.r, 1)
        res = cv2.warpAffine(image, M, (cols, rows))
        # Crop the image
        image = res[30:200, 90:230]
        image = cv2.resize(image, (self.h, self.w),interpolation = cv2.INTER_CUBIC)
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([155,25,0])
        upper = np.array([179,255,255])
        mask = cv2.inRange(image, lower, upper)
        result2 = cv2.bitwise_and(result, result, mask=mask)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(result2,(kernel_size, kernel_size),0)
        low_threshold = 0
        high_threshold = 0
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments
        line_image = np.copy(result2) * 0  # creating a blank to draw lines on
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for line in lines:
            for _x1,_y1,_x2,_y2 in line:
                x1.append(_x1)
                y1.append(_y1)
                x2.append(_x2)
                y2.append(_y2)
        
        return pd.DataFrame({'x1':x1,'y1':y1,'x2':x2,'y2':y2})

    def calibrar(self,diretorio:str):
        """Obtem os dados da planilha 'dados.xlsx' e salva em banco de dados para 
        uso interno da aplicação.
        

        Args:
            diretorio (str): caminho do arquivo no formato xlsx com os dados de calibragem.
        """
        calibragem = pd.read_excel(diretorio)
        dados = []
        leitura = []
        dif = []
        y = []
        x = []
        
        # Cria uma coluna com os valores de temperatura
        # Com base na coluna leituras da planilha
        n = 0
        for i in calibragem['leitura']:
            try:
                diferenca = i-calibragem['leitura'].iloc[n+1]
                for a in range(diferenca):
                    leitura.append(calibragem['leitura'].iloc[n]-a)
                n = n+1
                dif.append(diferenca)
            except:
                break
        # Cálcula os valores de y para cada linha da coluna leitura
        # Com base nos valores de y fornecidos na coluna de cada sensor
        dados.append(pd.Series(leitura,name='leitura'))
        for b in calibragem.keys():
            if b != 'leitura':
                print(b)
                for i in range(len(dif)):
                    yIn = calibragem[b].iloc[i]
                    vy = (calibragem[b].iloc[i+1] - yIn)/dif[i]
                    y.append(yIn)
                    for a in range(dif[i]-1):
                        y.append(yIn+vy)
                        yIn = yIn + vy
                dados.append(pd.Series(y,name=b))
                y = []
        # Salva tudo em um arquivo sqlite para uso interno da aplicação
        tabela = pd.concat(dados,axis=1)
        engine = create_engine('sqlite:///banco_de_dados.db')
        tabela.to_sql('calibragem', con=engine,if_exists='replace')
        # obtem da planilha a tebela de posição
        pd.read_excel(diretorio,sheet_name='posicao').to_sql('posicao', con=engine,if_exists='replace')
        
    def analizarImagens(self):
        """Salvas os dados extraidos das imagens no banco de dados interno da aplicação"""
        
        engine = create_engine('sqlite:///banco_de_dados.db', echo=False)
        fileNames = filedialog.askopenfilenames()
        print('Analizando imagens...')
        for i,cont in zip(fileNames,tqdm(range(len(fileNames)-1))):
            self.readImagem(i).to_sql(i, con=engine,if_exists='replace')
        
    def exportarResultados(self):
        engine = create_engine('sqlite:///banco_de_dados.db', echo=False)
        calibragem = pd.read_sql('calibragem', con=engine)
        posicao = pd.read_sql('posicao', con=engine)
        engine = create_engine('sqlite:///banco_de_dados.db', echo=False)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        arquivo = []
        data_ = []
        hora_ = []
        sensor_ = []
        nomes_sensor = []
        valores_lidos_ = pd.DataFrame()
        for i in posicao['sensor']:
            sensor_.append({i:[]})
            nomes_sensor.append(i)
        print('Compilando resultados...')
        for nomeTabela,z in zip(metadata.tables,tqdm(range(len(metadata.tables)-1))):
            if nomeTabela == 'calibragem' or nomeTabela == 'posicao':
                    continue
            arquivo.append(nomeTabela)
            year = datetime.date.today().year
            nome = nomeTabela
            nome = nome.split('/')
            nome = nome[len(nome)-1]
            data = f'{nome[10:12]}/{nome[13:14]}/{year}'
            hora = f'{nome[0:2]}:{nome[3:5]}:{nome[6:8]}'
            data_.append(data)
            hora_.append(hora)
            valor_lido_ = []
            for sensor,i in zip(posicao['sensor'],range(len(sensor_))):
                
                pIn = posicao['inicio'].loc[posicao['sensor']==sensor].values[0] # pixel inicial
                pOut = posicao['final'].loc[posicao['sensor']==sensor].values[0] # pixel final
                retas = pd.read_sql(nomeTabela,con=engine)
                retas = retas[['y2','x1']].loc[retas['x1'] > pIn]
                retas = retas[['y2','x1']].loc[retas['x1'] < pOut]
                numero = retas['y2'].min()
                calibragem['diferenca'] = abs(calibragem[sensor] - numero)
                calibragem = calibragem.sort_values('diferenca')
                valor_proximo = calibragem.iloc[0][sensor] # Valor do pixel mais próximo
                valor_lido = calibragem['leitura'].iloc[0] # valor da leitura mais próxima
                sensor_[i][sensor].append(valor_lido)
                
        for nomes,dado in zip(nomes_sensor,sensor_):
            valores_lidos_[nomes] = pd.DataFrame(dado)
            
        valores_lidos_['data'] = pd.DataFrame({'data':data_})
        valores_lidos_['hora'] = pd.DataFrame({'hora':hora_})
        valores_lidos_['arquivo'] = pd.DataFrame({'hora':arquivo})
            
        return valores_lidos_
        
        