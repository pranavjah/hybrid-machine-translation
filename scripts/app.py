from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route("/")
def index():
    # Renders the HTML template
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    # Get text from form submission or JSON
    text = request.form.get("text")  # Handles form data from the web interface
    if not text:
        return jsonify({"error": "No text provided for translation"}), 400

    # Perform translation
    tokens = tokenizer(text, return_tensors="pt")
    output = model.generate(**tokens)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Return the translation
    return jsonify({"translation": translation})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
