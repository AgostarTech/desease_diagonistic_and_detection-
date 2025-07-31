from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length

# Trim whitespace from inputs
def _strip(x):
    return x.strip() if x else x

class LoginForm(FlaskForm):
    username = StringField(
        'Username',
        validators=[DataRequired(), Length(min=3, max=80)],
        filters=[_strip]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(), Length(min=6, max=128)]
    )
    submit = SubmitField('Login')


class SignupForm(FlaskForm):
    username = StringField(
        'Username',
        validators=[DataRequired(), Length(min=3, max=25)],
        filters=[_strip]
    )
    email = StringField(
        'Email',
        validators=[DataRequired(), Email()],
        filters=[_strip]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(), Length(min=6)]
    )
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[DataRequired(), EqualTo('password', message='Passwords must match.')]
    )
    submit = SubmitField('Sign Up')
