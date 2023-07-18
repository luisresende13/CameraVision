from main import app

if __name__ == "__main__":
    print('MAIN EXECUTION START');
    app.run()
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))