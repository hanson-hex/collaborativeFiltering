 update [dbo].[t_AutoCoding_Field] set IsRequired = 1 where EntityUniqueName = 'publicCannedMessage' and name = 'message'
 update [dbo].[t_AutoCoding_Field] set IsRequired = 1 where EntityUniqueName = 'privateCannedMessage' and name = 'message'
 update [dbo].[t_AutoCoding_Field] set IsRequired = 1 where EntityUniqueName = 'privateCannedMessageCategory' and name = 'parent'
 update [dbo].[t_AutoCoding_Field] set IsRequired = 1 where EntityUniqueName = 'publicCannedMessageCategory' and name = 'parent'

 update [dbo].[t_AutoCoding_Entity] set label = 'Custom Agent Away Status', LabelForPlural = 'Custom Agent Away Statuses'  where  UniqueName = 'agentAwayStatus'
 update [dbo].[t_AutoCoding_Entity] set label = 'Custom Agent Away Status Config', LabelForPlural = 'Custom Agent Away Status Configs'  where  UniqueName = 'agentAwayStatusConfig'