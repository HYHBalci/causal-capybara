# Contributing to Causal Capybara

Thanks for helping improve Causal Capybara. Bug reports, documentation fixes, tests,
method adapters, and usability improvements are welcome.

## Before you start

- Search existing issues and pull requests. Open an issue before a large change so the
  intended behavior and statistical assumptions can be discussed.
- Follow the [development setup](README.md#the-app--windows-development-setup). The
  [installation guide](docs/INSTALL.md) distinguishes a release install from a
  source checkout.
- Keep real research data, project folders, credentials, and private reports out of
  the repository. Use generated or explicitly redistributable fixtures in tests.

## Submit a change

1. Make one focused change and explain the user problem it solves.
2. Add or update a meaningful test when behavior, statistical output, or a data
   contract changes. Document the estimator's assumptions and cite its method source.
3. Run the relevant Python or frontend tests described in [README](README.md#testing).
   For desktop changes, also run the Windows build and exercise the installed app.
4. Open a pull request with the behavior changed, checks run, and any limitations.

The project is licensed under [Apache-2.0](LICENSE). Under section 5, an intentional
contribution submitted for inclusion is licensed on those terms unless you explicitly
state otherwise. You retain the copyright to your contribution; no separate contributor
license agreement is required. Preserve third-party notices and identify material
changes to files when the license requires it.

Please report suspected security vulnerabilities through the private route in
[SECURITY.md](SECURITY.md), rather than a public issue.
