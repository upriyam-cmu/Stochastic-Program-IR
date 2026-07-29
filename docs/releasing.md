# Publishing releases

Releases use the dedicated `.github/workflows/release.yml` workflow and PyPI
Trusted Publishing. GitHub obtains short-lived credentials through OpenID
Connect; no PyPI API tokens belong in repository or environment secrets.

## One-time publisher setup

PyPI and TestPyPI use separate accounts and publisher registrations. Use the
[PyPI publisher settings](https://pypi.org/manage/account/publishing/) and
[TestPyPI publisher settings](https://test.pypi.org/manage/account/publishing/)
to register the following pending GitHub publishers before the first release:

| Index | Project | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- | --- |
| PyPI | `stoch-ir` | `upriyam-cmu` | `Stochastic-Program-IR` | `release.yml` | `pypi` |
| TestPyPI | `stoch-ir` | `upriyam-cmu` | `Stochastic-Program-IR` | `release.yml` | `testpypi` |

Create matching `pypi` and `testpypi` GitHub environments. Restrict both
environments to tags matching `v*`, and require approval from a repository
maintainer on `pypi`. TestPyPI does not require an approval because production
cannot run until its upload succeeds.

## Release sequence

1. Set the version in `pyproject.toml`, refresh `uv.lock`, and replace
   `Unreleased` in `CHANGELOG.md` with the release date.
2. Run the complete validation suite from `CONTRIBUTING.md`.
3. Merge the release commit into `main` and confirm all required checks pass.
4. Create an annotated tag whose name exactly matches `v<project-version>`,
   then push it:

   ```console
   git tag -a v0.1.0a1 -m "Release 0.1.0a1"
   git push origin v0.1.0a1
   ```

5. The release workflow builds and validates one wheel and source distribution,
   then uploads those exact artifacts to TestPyPI.
6. Inspect the TestPyPI project and test installation in a clean environment:

   ```console
   python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     "stoch-ir==0.1.0a1"
   ```

7. Approve the waiting `pypi` environment deployment. The workflow publishes
   the previously tested artifacts to PyPI with digital attestations.
8. Create the corresponding GitHub release from the existing tag.

Published filenames and versions cannot be replaced. Fix any release problem
with a new version rather than moving or recreating its tag.
