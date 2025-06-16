from core.train import train
from core.test import test
from core.util import create_label


tick= 'FRT'
train(f'{tick}_train.csv')
test(f'{tick}_test.csv','model_'+tick+create_label()+'.keras')
