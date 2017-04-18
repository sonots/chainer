How to Create your Extension
--------------------------

:class:`~chainer.training.Extension` is a useful functionality for
:class:`~chainer.training.Trainer` to customize your training loop. Adding an
:class:`~chainer.training.Extension` object to your
:class:`~chainer.training.Trainer` object using
:meth:`~chainer.training.Trainer.extend` enables to launch an arbitrary
function that takes the :class:`~chainer.training.Trainer` object as an input
argument during the training process at given intervals.

There are two ways to create your own :class:`~chainer.training.Extension`
object.

1. Write a class inherited from :class:`~chainer.training.Extension`
2. Decorate your function with :meth:`~chainer.training.make_extension`

Let's start to write an extension that performs learning rate dropping.

Option 1: Create a decorated function
````````````````````````````````````````````````````````````````````````````

.. doctest::

    @chainer.training.make_extension(trigger=(1000, 'iteration')):
    def learning_rate_dropping(trainer):
        trainer.updater.get_optimizer('main').lr *= 0.1

    trainer.extend(learning_rate_dropping)
