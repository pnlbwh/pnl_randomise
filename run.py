from flask import Flask, render_template
import getpass

app = Flask(__name__)

# @app.route("/")
# def template_test():
    # return render_template(
            # 'template.html',
            # my_string='Wheee!',
            # title='home',
            # my_list=[0,1,2,3,4,5])


@app.route("/")
def web(corrp_map_classes, df):
    user_name = getpass.getuser()

    # app.run(debug=True)
    return render_template(
            'template.html',
            my_string='Wheee!',
            title='home',
            my_list=[0,1,2,3,4,5])


if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(debug=True)
