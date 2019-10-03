### A Pytorch example on adversarial networks to defeat systematic uncertainties

This is an example on how, using adversarial training,
it is possible to make a neural network unsensitive wrt
to systematic uncertainties that could affect your measurement.

The approach has been already demonstrated effective by
a paper published in 2016, `Learning to pivot with adversarial
networks` (https://arxiv.org/abs/1611.01046) - not written by me,
to be clear.

Systematic uncertainties can come from many sources. In image
processing, for example, the distance at which each picture has been
taken could constitute an important uncertainty to take into account.
To make your model, and predictions, robust you can follow two approaches:
either you take all your pictures at the same distance (think at the
airport face scanners, where they force you to stay in a certain spot),
or you include pictures taken at any possible distance. However,
"any" could mean a lot. What you could also do is to force your network
to learn exclusively features that are unsensitive to the distance at
which the picture has been taken. You can do this by using adversarial
networks.

In more general terms, the following study shows how including
systematic variations in the training is often not enough to
create a robust model, which in principle should be unsensitive
wrt to such systematic errors. Adversarial training solves the problem:
in such scenario a discriminant is trained to distinguish between the
classes of interest, while an adversarial term learns to distinguish
between the nominal sample and the systematic variations based on the
output of the first classifier.

The two components are trained simultaneously using a loss function
that combines the losses of the two terms. In particular, the loss
of the adversarial term is multiplied by a (large) penalty and
subtracted from the loss of the discriminant so to give "positive"
points to a discriminant that can distinguish btw classes and
"negative" points to a discriminant that produces very different
distribution for the nominal sample and the systematic uncertainties.

#### How to run:
The code can be run either as a Jupyter notebook

```
jupyter lab main.ipynb
```

or converted to slides

```
jupyter nbconvert main.ipynb --to slides --post serve
--SlidesExporter.reveal_theme=serif
--SlidesExporter.reveal_scroll=True
--TagRemovePreprocessor.remove_input_tags='{"to_remove"}'
--TagRemovePreprocessor.remove_output_tags='{"to_remove_out"}'
--no-prompt
```

In this second case, you also need to copy the `custom.css` file 
into `~/.jupyter/custom/custom.css`.

The required libraries are listed in the `requirements.txt` file.
Install them with:

```
pip install -r requirements.txt
```
