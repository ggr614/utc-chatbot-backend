"""Tests for create-admin CLI command."""

import pytest
from unittest.mock import MagicMock, Mock, patch


class TestCreateAdmin:
    """Test the create-admin CLI command."""

    @patch("builtins.input", return_value="y")
    @patch("getpass.getpass", side_effect=["securepass123", "securepass123"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_create_new_admin(
        self, mock_settings, mock_client_cls, mock_getpass, mock_input
    ):
        from main import command_create_admin
        import argparse

        # Setup mocks
        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_client.user_exists.return_value = False
        mock_client.create_user.return_value = "new-uuid"
        mock_client_cls.return_value = mock_client

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 0
        mock_client.create_user.assert_called_once()

    @patch("getpass.getpass", side_effect=["password1", "password2"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_password_mismatch_returns_error(
        self, mock_settings, mock_client_cls, mock_getpass
    ):
        from main import command_create_admin
        import argparse

        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 1  # Should fail due to mismatch

    @patch("builtins.input", return_value="y")
    @patch("getpass.getpass", side_effect=["newpass123", "newpass123"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_update_existing_admin(
        self, mock_settings, mock_client_cls, mock_getpass, mock_input
    ):
        from main import command_create_admin
        import argparse

        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_client.user_exists.return_value = True
        mock_client.update_password.return_value = True
        mock_client_cls.return_value = mock_client

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 0
        mock_client.update_password.assert_called_once()
