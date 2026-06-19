code = f"""
<div style='font-family:Inter,sans-serif; min-width:200px;'>
<h4 style='color:#1e3a5f; margin:0;'>Hotspot Cluster {int(row['cluster_id'])}</h4>
<hr style='margin:4px 0;'>
<b>Severity:</b> <span style='color:{colour};font-weight:700;'>{severity}</span><br>
<b>Violations:</b> {count:,}<br>
<b>Severity Score:</b> {score:.1f}/100<br>
<b>Junction:</b> {junction}<br>
<b>Unique Vehicles:</b> {int(row.get('unique_vehicles', 0)):,}
</div>
"""
