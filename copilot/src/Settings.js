import React, { useState, useEffect } from 'react';
import { TextField, Button, Select, MenuItem, Chip, Box, FormControl, InputLabel } from '@mui/material';
import axios from 'axios';

function Settings({ onClearChat }) {
    const [settings, setSettings] = useState({
        default_portfolio_value: 100000,
        risk_profile: 'moderate',
        preferred_sectors: [],
        notification_frequency: 'daily',
        theme: 'light',
        language: 'fr'
    });

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(process.env.REACT_APP_API_URL + '/settings', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            setSettings(response.data);
        } catch (error) {
            console.error("Error fetching settings:", error);
        }
    };

    const handleSettingsChange = (setting, value) => {
        setSettings(prevSettings => ({ ...prevSettings, [setting]: value }));
    };

    const clearChat = async () => {
        try {
            const token = localStorage.getItem('token');
            await axios.post(process.env.REACT_APP_API_URL + '/clear_chat', {}, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            alert('Historique du chat effacé avec succès');
            onClearChat();
        } catch (error) {
            console.error('Erreur lors de l\'effacement de l\'historique du chat:', error);
            alert('Erreur lors de l\'effacement de l\'historique du chat. Veuillez réessayer.');
        }
    };

    const saveSettings = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.post(process.env.REACT_APP_API_URL + '/settings', settings, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.data.message === "Paramètres mis à jour avec succès") {
                alert('Paramètres sauvegardés avec succès');
            } else {
                throw new Error('Réponse inattendue');
            }
        } catch (error) {
            console.error('Erreur lors de la sauvegarde des paramètres:', error);
            alert('Erreur lors de la sauvegarde des paramètres. Veuillez réessayer.');
        }
    };

    return (
        <Box>
            <TextField
                label="Valeur par défaut du portefeuille"
                value={settings.default_portfolio_value}
                onChange={(e) => handleSettingsChange('default_portfolio_value', e.target.value)}
                type="number"
                fullWidth
                margin="normal"
            />
            <FormControl fullWidth margin="normal">
                <InputLabel>Profil de risque</InputLabel>
                <Select
                    value={settings.risk_profile}
                    onChange={(e) => handleSettingsChange('risk_profile', e.target.value)}
                >
                    <MenuItem value="low">Faible</MenuItem>
                    <MenuItem value="moderate">Modéré</MenuItem>
                    <MenuItem value="high">Élevé</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Secteurs préférés</InputLabel>
                <Select
                    multiple
                    value={settings.preferred_sectors}
                    onChange={(e) => handleSettingsChange('preferred_sectors', e.target.value)}
                    renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {selected.map((value) => (
                                <Chip key={value} label={value} />
                            ))}
                        </Box>
                    )}
                >
                    <MenuItem value="Technology">Technologie</MenuItem>
                    <MenuItem value="Healthcare">Santé</MenuItem>
                    <MenuItem value="Finance">Finance</MenuItem>
                    <MenuItem value="Energy">Énergie</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Fréquence des notifications</InputLabel>
                <Select
                    value={settings.notification_frequency}
                    onChange={(e) => handleSettingsChange('notification_frequency', e.target.value)}
                >
                    <MenuItem value="daily">Quotidienne</MenuItem>
                    <MenuItem value="weekly">Hebdomadaire</MenuItem>
                    <MenuItem value="monthly">Mensuelle</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Thème</InputLabel>
                <Select
                    value={settings.theme}
                    onChange={(e) => handleSettingsChange('theme', e.target.value)}
                >
                    <MenuItem value="light">Clair</MenuItem>
                    <MenuItem value="dark">Sombre</MenuItem>
                </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
                <InputLabel>Langue</InputLabel>
                <Select
                    value={settings.language}
                    onChange={(e) => handleSettingsChange('language', e.target.value)}
                >
                    <MenuItem value="fr">Français</MenuItem>
                    <MenuItem value="en">Anglais</MenuItem>
                </Select>
            </FormControl>
            <Button
                onClick={clearChat}
                variant="contained"
                color="secondary"
                style={{ marginTop: '20px', marginRight: '10px' }}
            >
                Effacer l'historique du chat
            </Button>
            <Button 
                onClick={saveSettings} 
                variant="contained" 
                color="primary"
                style={{ marginTop: '20px' }}
            >
                Sauvegarder les paramètres
            </Button>
        </Box>
    );
}

export default Settings;