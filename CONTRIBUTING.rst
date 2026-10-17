============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given. 

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/mwv/cdtw/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

cdtw could always use more documentation, whether as part of the 
official cdtw docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/mwv/cdtw/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `cdtw` for local development.

1. Fork the `cdtw` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/cdtw.git

3. Install your local copy using uv (requires Rust 1.85 or newer, see
   https://rustup.rs; ``uv sync`` builds the extension with maturin, and
   rebuilds it whenever Rust sources change)::

    $ cd cdtw/
    $ uv sync

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature
   
   Now you can make your changes locally.

5. When you're done making changes, check that the tests pass::

    $ cargo test -p cdtw-core
    $ uv run pytest tests/

   For performance changes, compare benchmark runs from before and after
   (see the Benchmarks section of README.rst).

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work on Python 3.10+. Check the GitHub Actions
   CI run on the pull request and make sure the tests pass.

Tips
----

To run a subset of tests::

	$ uv run pytest tests/test_dtw.py -v