from flask import Flask, render_template, request, redirect, jsonify
from sklearn.decomposition import PCA
import word2vec
import hashlib
import colorsys
import numpy as np
import matplotlib.cm as cm

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


class DocumentStore:
    def __init__(self):
        self._document_list = []
        self._model = {}
        self._words = []
        self._fcm = None
        self._vec = None
        self._key = None

    def store(self, document):
        # s = hash(document)
        if not self.existOf(document):
            print("documentの追加")
            self._document_list.append(document)
            # self._model[self.hash(self.get_document())] = None
            # print(self._model)
            self._key = self.hash(self.get_document())
            print(self._key)
        else:
            print("documentは追加されませんでした")

    def update_model(self, model, words, fcm):
        # self._model = model
        self._model[self._key] = model
        self._words = words
        self._fcm = fcm
        print(self._model)

    def update_vec_with_word(self, vector):
        self._vec = vector
    
    def update_pca(self, pca):
        self._pca = pca
    def get_pca(self):
        return self._pca

    def get_vec_with_word(self):
        return self._vec

    def get_words(self):
        return self._words

    def get_model(self):
        # return self._model
        if not self._key in self._model.keys():
            return None
        return self._model[self._key]

    def get_document(self):
        if len(self._document_list) > 1:
            return "\n".join(self._document_list)
        else:
            return self.get_near_data()

    def get_fcm(self):
        return self._fcm

    def get_near_data(self):
        if self.isEmpty():
            return None
        else:
            return self._document_list[-1]

    # def append_key(d1, d2):
    def hash(self, d):
        s = "".join(d)
        s = hashlib.sha256(s.encode()).hexdigest()
        return s

    def existOf(self, document):
        return document in self._document_list

    def isEmpty(self):
        return len(self._document_list) == 0


# BUFFER = Buffer()
document_store = DocumentStore()


@app.route("/")
def index():
    return render_template("index.html", document=None)


@app.route("/action", methods=["POST"])
def action():
    documents = request.form["documents"]
    if documents is not "":
        document_store.store(documents)
        return redirect("/document")
    else:
        pass
    return redirect("/")


@app.route("/document")
def document():
    # document = BUFFER.get_near_data()
    document = document_store.get_near_data()  # 最近入力した文書を取り出す
    if document is None:
        return redirect("/")
    documents = [sentence for sentence in document.split(
        "\r\n") if sentence is not ""]
    return render_template("document.html", documents=documents)


@app.route("/refresh")
def reflesh():
    global document_store
    document_store = DocumentStore()
    print("データを消しました")
    return redirect("/")


@app.route("/word2vec")
def vector():

    if document_store.get_near_data() is None:
        return redirect("/")
    if document_store.get_model() is None:
        print("モデルがありませんでした")
        model = word2vec.w2c(document_store.get_document())
        words = word2vec.word_in_document(document_store.get_document())

        pca = PCA(n_components=2)
        press_vector = []
        pca.fit(model.wv.vectors)
        press_vector = pca.fit_transform(model.wv.vectors)

        clustering_result = word2vec.fcm(10, model.wv.vectors)

        document_store.update_model(model, words, clustering_result)
        document_store.update_pca(press_vector)
        print("学習しました")
    else:
        print("モデルが有りました")
        model = document_store.get_model()
        press_vector = document_store.get_pca()
        clustering_result = document_store.get_fcm()  
        print("モデルを読み込みました")

    color = draw_colors(10)

    wordvectors = []

    for i, word in enumerate(document_store.get_words()):
        label = clustering_result["label"][i]
        c = color[label]
        wordvectors.append(
            {
                "label": word,
                "data": {
                    "x": float(press_vector[i][0]),
                    "y": float(press_vector[i][1])
                },
                "color": rgba(c)
            }
        )

    # wordvectors = [{"label": word, "data": {"x": float(press_vector[i][0]),
    #                                         "y": float(press_vector[i][1])}}for i, word in enumerate(model.wv.vocab.keys())]
    # wordvectors = [{"label": word, "data": {"x": float(press_vector[i][0]),
    #                                         "y": float(press_vector[i][1])}}for i, word in enumerate(document_store.get_words())]

    document_store.update_vec_with_word(wordvectors)
    return jsonify({
        # "vector": wordvector,
        "dict": wordvectors,
    })


def draw_colors(n_colors):
    colors = np.array([[0, 255, 0, 1],
                       [102, 0, 255, 1],
                       [102, 255, 255, 1],
                       [204, 51, 0, 1],
                       [255, 255, 0, 1],
                       [0, 153, 153, 1],
                       [204, 153, 255, 1],
                       [153, 0, 0, 1],
                       [255, 153, 0, 1],
                       [255, 204, 204, 1]])
    return colors


def rgba(color):
    return "rgba({},{},{},{})".format(color[0], color[1], color[2], color[3])


if __name__ == "__main__":
    app.run(debug=True)
