import sys
from gmm import save_model, predict
from seg import vadSeg, segment

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('help: python main.py [task] [model|wav]')
        exit()

    task = sys.argv[1]

    if task == 'train':
        speech = sys.argv[2]
        save_model(speech.replace('.wav', '.mdl'), speech)
        print('done.')

    elif task == 'seg':
        model = sys.argv[2]
        dialogue = sys.argv[3]
        vadSeg(model, dialogue)
        print('done.')

    elif task == 'verify':
        model = sys.argv[2]
        speech = sys.argv[3]
        score = predict(model, speech)
        print(score)
        if score > -50:
            print('yes')
        else:
            print('no')
