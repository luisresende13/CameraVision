from main import app
# import os

if __name__ == "__main__":
    print('wsgi.py __main__ EXECUTING...')
    app.run()
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
